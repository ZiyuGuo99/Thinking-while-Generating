"""
TwiG evaluation on T2I-Compbench++
"""

import argparse
import os
import re

import numpy as np
import PIL.Image
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images

def parse_args():
    parser = argparse.ArgumentParser(description="TwiG Evaluation")
    parser.add_argument("--idx", type=int, default=0, help="GPU index (worker rank)")
    parser.add_argument("--num_workers", type=int, default=8, help="Total number of workers")
    parser.add_argument("--dataset", type=str, default="", help="Dataset path")
    parser.add_argument("--model_path", type=str, default="", help="Model path")
    parser.add_argument("--output_path", type=str, required=True, help="Output path")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--parallel_size", type=int, default=1, help="Parallel batch size")
    parser.add_argument("--cfg_weight", type=float, default=5.0, help="CFG weight")
    parser.add_argument("--image_token_num", type=int, default=576, help="Image tokens per image")
    parser.add_argument("--num_layers", type=int, default=3, help="Number of horizontal layers")
    parser.add_argument("--img_size", type=int, default=384, help="Image size")
    parser.add_argument("--patch_size", type=int, default=16, help="Patch size")
    parser.add_argument("--decode_channels", type=int, default=8, help="Channel dim")
    parser.add_argument("--theta", type=int, default=90, help="Reflection pass threshold")
    return parser.parse_args()

@torch.inference_mode()
def reflection(
    theta,
    ori_prompt,
    step,
    prev_description,
    description,
    image,
    vl_gpt=None,
    tokenizer=None,
    vl_chat_processor=None,
):
    sections = ["top third", "middle third", "bottom third"]
    section = sections[step]

    prompt_text = f"""This image was generated in 3 layers. Analyze the image generation quality focusing on layer {step+1} and optimize its description.

Original prompt: {ori_prompt}{prev_description}
Target Layer {step+1} description: {description}
Target section: Layer {step+1}

IMPORTANT CONTEXT:
- The image is square. (For layer 1: Top 1/3 contains actual content, bottom 2/3 is placeholder (beige mask tokens); For layer 2: Top 2/3 contains actual content, bottom 1/3 is placeholder; For layer 3: Full image with complete content)
- Focus ONLY on evaluating the {section} of the actual content
- Ignore placeholder regions in your evaluation

YOUR TASK:
1. Score (0-100): How well does the {section} match the prompt and execute the description? Note: this layer may only show partial content, not full prompt alignment needed.
  Consider: color accuracy, object completeness, detail richness, spatial relationships, visual coherence
  Scoring rules:
  - Use the full integer scale from 0 to 100 (e.g., 83, 91, 76).
  - DO NOT round to the nearest 5 or 10. (e.g., Do not just give 80, 85, 90).
  - Be critical. A score of 95+ is reserved for perfection.

2. Provide an improved description for generating ONLY this {section}.
  The improved description should:
  - Address specific quality issues you identified
  - Add precise visual details (colors, textures, positions, sizes)
  - Maintain spatial accuracy for this horizontal section only
  - Use concrete, literal descriptions without vague terms
  - Ensure consistency with the overall prompt intent
  - Consider harmony with future layers for the first two layers, othervise ensure it complements the existing upper layers
  - Do not describe other sections or use positional words like "top/middle/bottom"
  - Use concrete language, avoid "section/layer/portion" words
  - Stay faithful to original prompt - Do not invent elements not implied by the original prompt, and coordinate across all three layer descriptions to ensure you don't omit any objects that appear in the original prompt, making sure they appear in the appropriate layer at the right time.

Output format:
SCORE: [0-100, NOT a multiple of 5 unless coincidental]
IMPROVED_DESCRIPTION: [Single sentence describing only the {section} with enhanced precision, faithful to prompt]"""

    conversation = [
        {"role": "<|User|>", "content": f"<image_placeholder>\n{prompt_text}", "images": [image]},
        {"role": "<|Assistant|>", "content": ""},
    ]
    prepare_inputs = vl_chat_processor(
        conversations=conversation, images=load_pil_images(conversation), force_batchify=True
    ).to(vl_gpt.device)
    inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

    outputs = vl_gpt.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=512,
        do_sample=False,
        use_cache=True,
    )
    answer = (
        tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        .replace("User: ", "")
        .replace("Assistant: ", "")
    )

    try:
        score_match = re.search(r"SCORE:\s*(\d+)", answer)
        desc_match = re.search(r"IMPROVED_DESCRIPTION:\s*(.+)", answer, re.DOTALL)
        score = int(score_match.group(1)) if score_match else 100
        improved_desc = desc_match.group(1).strip() if desc_match else ""
        score = 100 if improved_desc == "" else score
    except Exception as e:
        print(f"[Warning] Reflection parsing failed for layer {step+1}: {e}")
        score, improved_desc = 100, ""

    return score >= theta, improved_desc


@torch.inference_mode()
def understand(
    t2i_prompt,
    image=None,
    previous=None,
    index=-1,
    vl_gpt=None,
    tokenizer=None,
    vl_chat_processor=None,
):
    if image:
        prompt_text = f'You are describing an image in 3 horizontal parts, from top to bottom. The original prompt is: "{t2i_prompt}". The following description(s) have already been written: "{previous}". A partial image has been generated up to this point and is attached. Now write the next sentence, describing only the new visible content that would naturally continue from the generated part. Only use details that are visible in the image or clearly implied by the original prompt. Do not invent new objects or textures. Do not repeat previous content. Do not use any labels like \'Part\', \'Top\', or \'Layer\'. Keep the language plain and literal. Output only the new sentence. If there is any conflict between the image and the previous text, follow the visual content.'
        conversation = [
            {"role": "<|User|>", "content": f"<image_placeholder>\n{prompt_text}", "images": [image]},
            {"role": "<|Assistant|>", "content": ""},
        ]
        pil_images = load_pil_images(conversation)
        prepare_inputs = vl_chat_processor(
            conversations=conversation, images=pil_images, force_batchify=True
        ).to(vl_gpt.device)
        inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)
        outputs = vl_gpt.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=512,
            do_sample=False,
            use_cache=True,
        )
    else:
        prompt_text = f'Expand the image prompt "{t2i_prompt}" into a description of the first of three horizontal visual sections. Only describe visible elements exactly as stated or implied in the prompt. Do not invent any new objects, details, or backgrounds. Do not add section labels like \'Part 1\' or \'Top\'. Keep the description short and literal. Output only the sentence.'
        conversation = [
            {"role": "<|User|>", "content": prompt_text},
            {"role": "<|Assistant|>", "content": ""},
        ]
        prompt = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conversation,
            sft_format=vl_chat_processor.sft_format,
            system_prompt="",
        )
        input_ids = vl_chat_processor.tokenizer.encode(prompt)
        input_ids = torch.LongTensor(input_ids).to(vl_gpt.device)
        inputs_embeds = vl_gpt.language_model.get_input_embeddings()(input_ids).unsqueeze(0)
        outputs = vl_gpt.language_model.generate(
            inputs_embeds=inputs_embeds,
            pad_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=512,
            do_sample=False,
            use_cache=True,
        )
    answer = (
        tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        .replace("User: ", "")
        .replace("Assistant: ", "")
    )
    return answer


@torch.inference_mode()
def generate(
    args,
    foldername,
    prompt: str,
    mmgpt: MultiModalityCausalLM,
    vl_chat_processor: VLChatProcessor,
    temperature: float = 1,
    parallel_size: int = 1,
    cfg_weight: float = 5,
    image_token_num_per_image: int = 576,
    num_layers: int = 3,
    img_size: int = 384,
    patch_size: int = 16,
    decode_channels: int = 8,
    name: str = None,
):
    # Pre-create all necessary directories at the start
    for sub_dir in ["cache", "cache_step0", "cache_step1", "cache_step2", "samples", "descriptions"]:
        os.makedirs(os.path.join(foldername, sub_dir), exist_ok=True)

    reflection_enabled = True
    regenerate_pending = False
    tokens_per_layer = image_token_num_per_image // num_layers
    all_insert_num = [k * tokens_per_layer - 1 for k in range(1, num_layers)]

    new_description = understand(
        t2i_prompt=prompt,
        vl_gpt=mmgpt,
        tokenizer=vl_chat_processor.tokenizer,
        vl_chat_processor=vl_chat_processor,
    )
    model_input = f"User: {new_description}\n\nAssistant:<begin_of_image>"
    input_ids = vl_chat_processor.tokenizer.encode(model_input)
    input_ids = torch.LongTensor(input_ids)

    tokens = torch.zeros((parallel_size * 2, len(input_ids)), dtype=torch.int).cuda()
    for i in range(parallel_size * 2):
        tokens[i, :] = input_ids
        if i % 2 != 0:
            tokens[i, 1:-1] = vl_chat_processor.pad_id

    inputs_embeds = mmgpt.language_model.get_input_embeddings()(tokens)
    generated_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.int).cuda()

    def get_insert_embeds(extra_text):
        insert_ids = vl_chat_processor.tokenizer.encode(extra_text)
        insert_ids = torch.LongTensor(insert_ids)
        insert_tokens = torch.zeros((parallel_size * 2, len(insert_ids)), dtype=torch.int).cuda()
        for j in range(parallel_size * 2):
            insert_tokens[j, :] = insert_ids
            if j % 2 != 0:
                insert_tokens[j, 1:-1] = vl_chat_processor.pad_id
        return mmgpt.language_model.get_input_embeddings()(insert_tokens)

    # Dry up the decoding and saving of images
    def decode_and_save_image(save_path, batch_idx=0):
        dec = mmgpt.gen_vision_model.decode_code(
            generated_tokens.to(dtype=torch.int),
            shape=[parallel_size, decode_channels, img_size // patch_size, img_size // patch_size],
        )
        dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
        dec = np.clip((dec + 1) / 2 * 255, 0, 255).astype(np.uint8)
        PIL.Image.fromarray(dec[batch_idx]).save(save_path)
        return dec

    gen_image_embed = []
    previous_description = model_input
    all_descriptions = [previous_description.replace("\n\nAssistant:<begin_of_image>", "")]

    i = 0
    while i < image_token_num_per_image:
        if i - 1 in all_insert_num or i == 0 or (
            i == image_token_num_per_image - 1 and regenerate_pending
        ):
            if i != 0:
                del outputs.past_key_values
                del outputs
            outputs = mmgpt.language_model.model(
                inputs_embeds=inputs_embeds,
                use_cache=True,
                past_key_values=None,
            )
        else:
            outputs = mmgpt.language_model.model(
                inputs_embeds=img_embeds.unsqueeze(dim=1),
                use_cache=True,
                past_key_values=past_key_values,
            )
        past_key_values = outputs.past_key_values
        hidden_states = outputs.last_hidden_state

        logits = mmgpt.gen_head(hidden_states[:, -1, :])
        logit_cond = logits[0::2, :]
        logit_uncond = logits[1::2, :]
        logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
        probs = torch.softmax(logits / temperature, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated_tokens[:, i] = next_token.squeeze(dim=-1)
        next_token = torch.cat(
            [next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1
        ).view(-1)
        img_embeds = mmgpt.prepare_gen_img_embeds(next_token)
        inputs_embeds = torch.cat([inputs_embeds, img_embeds.unsqueeze(dim=1)], dim=1)
        gen_image_embed.append(img_embeds)

        if i in all_insert_num:
            index = (i + 1) // tokens_per_layer
            save_path = os.path.join(foldername, "cache", name + ".png")
            decode_and_save_image(save_path)

            if i == tokens_per_layer - 1 and reflection_enabled and not regenerate_pending:
                decode_and_save_image(os.path.join(foldername, "cache_step0", name + ".png"))
            elif i == 2 * tokens_per_layer - 1 and reflection_enabled and not regenerate_pending:
                decode_and_save_image(os.path.join(foldername, "cache_step1", name + ".png"))

            pre_ref = (
                "\nPrevious layers' description: " + all_descriptions[-2].replace("User: ", "") + "\n"
                if len(all_descriptions) > 1
                else ""
            )

            if reflection_enabled and not regenerate_pending:
                accept, improved_desc = reflection(
                    args.theta,
                    ori_prompt=prompt,
                    step=index - 1,
                    prev_description=pre_ref,
                    description=new_description,
                    image=save_path,
                    vl_gpt=mmgpt,
                    tokenizer=vl_chat_processor.tokenizer,
                    vl_chat_processor=vl_chat_processor,
                )
                if not accept:
                    insert = (
                        all_descriptions[-2] + " " + improved_desc
                        if len(all_descriptions) > 1
                        else "User: " + improved_desc
                    )
                    previous_description = insert
                    all_descriptions[-1] = insert.replace(
                        "\n\nAssistant:<begin_of_image>", ""
                    ).replace("  ", "")
                    insert_embeds = get_insert_embeds(f"{insert}\n\nAssistant:<begin_of_image>")
                    gen_image_embed = gen_image_embed[:-tokens_per_layer]
                    inputs_embeds = (
                        torch.cat([insert_embeds, torch.stack(gen_image_embed, dim=1)], dim=1)
                        if len(gen_image_embed) != 0
                        else insert_embeds
                    )
                    regenerate_pending = True
                    i -= tokens_per_layer
                    i += 1
                    continue
            elif regenerate_pending:
                regenerate_pending = False

            new_description = understand(
                t2i_prompt=prompt,
                image=save_path,
                previous=previous_description,
                index=index + 1,
                vl_gpt=mmgpt,
                tokenizer=vl_chat_processor.tokenizer,
                vl_chat_processor=vl_chat_processor,
            )
            insert = (
                previous_description.replace("\n\nAssistant:<begin_of_image>", "")
                + " "
                + new_description
            )
            previous_description = insert
            all_descriptions.append(insert.replace("\n\nAssistant:<begin_of_image>", ""))

            insert_embeds = get_insert_embeds(f"{insert}\n\nAssistant:<begin_of_image>")
            inputs_embeds = torch.cat([insert_embeds, torch.stack(gen_image_embed, dim=1)], dim=1)

        if i == image_token_num_per_image - 1 and not regenerate_pending:
            index = (i + 1) // tokens_per_layer
            save_path = os.path.join(foldername, "cache_step2", name + ".png")
            decode_and_save_image(save_path)

            pre_ref = (
                "\nPrevious layers' description: \n" + all_descriptions[-2].replace("User: ", "")
                if len(all_descriptions) > 1
                else ""
            )
            accept, improved_desc = reflection(
                args.theta,
                ori_prompt=prompt,
                step=index - 1,
                prev_description=pre_ref,
                description=new_description,
                image=save_path,
                vl_gpt=mmgpt,
                tokenizer=vl_chat_processor.tokenizer,
                vl_chat_processor=vl_chat_processor,
            )
            if not accept:
                insert = (
                    all_descriptions[-2] + " " + improved_desc
                    if len(all_descriptions) > 1
                    else "User: " + improved_desc
                )
                previous_description = insert
                all_descriptions[-1] = insert.replace(
                    "\n\nAssistant:<begin_of_image>", ""
                ).replace("  ", "")
                insert_embeds = get_insert_embeds(f"{insert}\n\nAssistant:<begin_of_image>")
                gen_image_embed = gen_image_embed[:-tokens_per_layer]
                inputs_embeds = (
                    torch.cat([insert_embeds, torch.stack(gen_image_embed, dim=1)], dim=1)
                    if len(gen_image_embed) != 0
                    else insert_embeds
                )
                regenerate_pending = True
                i -= tokens_per_layer
                i += 1
                continue

        i += 1

    # Save final samples for each item in parallel batch
    for j in range(parallel_size):
        save_path = os.path.join(foldername, "samples", name + ".png")
        decode_and_save_image(save_path, batch_idx=j)

    with open(os.path.join(foldername, "descriptions", f"{name}.txt"), "w") as f:
        for idx, desc in enumerate(all_descriptions):
            f.write(f"Step {idx}:\n{desc}\n\n")

def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.idx)

    with open(args.dataset, "r") as f:
        lines = f.readlines()

    model_path = args.model_path
    vl_chat_processor = VLChatProcessor.from_pretrained(model_path)
    tokenizer = vl_chat_processor.tokenizer
    vl_gpt = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

    os.makedirs(args.output_path, exist_ok=True)

    # Split dataset across num_workers (must match n in eval.sh).
    chunk_size = len(lines) // args.num_workers + 1
    start_idx = args.idx * chunk_size
    end_idx = (args.idx + 1) * chunk_size
    process_lines = lines[start_idx:end_idx]

    for idx, line in enumerate(tqdm(process_lines, total=len(process_lines))):
        i = start_idx + idx
        generate(
            args,
            args.output_path,
            line.strip(),
            vl_gpt,
            vl_chat_processor,
            temperature=args.temperature,
            parallel_size=args.parallel_size,
            cfg_weight=args.cfg_weight,
            image_token_num_per_image=args.image_token_num,
            num_layers=args.num_layers,
            img_size=args.img_size,
            patch_size=args.patch_size,
            decode_channels=args.decode_channels,
            name=f"{line.strip()}_{i:03d}",
        )

if __name__ == "__main__":
    main()