import os
import sys


# print('[System ARGV] ' + str(sys.argv))

root = os.path.dirname(os.path.abspath(__file__))
backend_path = os.path.join(root, 'backend', 'headless')
sys.path += [root, backend_path]

# os.chdir(root)
# os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
# os.environ["GRADIO_SERVER_PORT"] = "7865"

import random
import time
import modules.config
import modules.html
import modules.async_worker as worker
import modules.constants as constants
import modules.flags as flags
import modules.gradio_hijack as grh
import modules.advanced_parameters as advanced_parameters
import argparse
import backend.headless.fcbh.model_management as model_management

def generate_image(args):
    with model_management.interrupt_processing_mutex:
        model_management.interrupt_processing = False

    execution_start_time = time.perf_counter()
    task = worker.AsyncTask(args=args)
    worker.async_tasks.append(task)
    while True:
        time.sleep(0.01)
        if len(task.yields) > 0:
            flag, product = task.yields.pop(0)
            if flag == 'preview':
                continue
            if flag == 'results':
                continue
            if flag == 'finish':
                break

    execution_time = time.perf_counter() - execution_start_time
    print(f'Total time: {execution_time:.2f} seconds')
    return


def refresh_seed(r, seed_string):
    if r:
        return random.randint(constants.MIN_SEED, constants.MAX_SEED)
    else:
        try:
            seed_value = int(seed_string)
            if constants.MIN_SEED <= seed_value <= constants.MAX_SEED:
                return seed_value
        except ValueError:
            pass
        return random.randint(constants.MIN_SEED, constants.MAX_SEED)

  
def ctrls(prompt: str, negative_prompt: str, image_seed: str, seed_random: bool) -> list:
    style_selections=modules.config.default_styles
    performance_selection=modules.config.default_performance
    aspect_ratios_selection=modules.config.default_aspect_ratio
    image_number=modules.config.default_image_number
    image_seed = refresh_seed(seed_random, image_seed)
    sharpness=modules.config.default_sample_sharpness
    guidance_scale=modules.config.default_cfg_scale
    base_model=modules.config.default_base_model_name
    refiner_model=modules.config.default_refiner_model_name
    refiner_switch=modules.config.default_refiner_switch
    lora_ctrls=[item for row in modules.config.default_loras for item in row]
    input_image_checkbox=False
    current_tab='uov'
    uov_method=flags.disabled
    uov_input_image=grh.Image()
    outpaint_selections=[]
    inpaint_input_image=grh.Image()
    inpaint_additional_prompt=""
    default_end, default_weight = flags.default_parameters[flags.default_ip]
    ip_ctrls=[]
    for _ in range(4):
        ip_ctrls.append(grh.Image())
        ip_ctrls.append(default_end)
        ip_ctrls.append(default_weight)
        ip_ctrls.append(flags.default_ip)

    ctrls = [
        prompt, negative_prompt, style_selections,
        performance_selection, aspect_ratios_selection, image_number, image_seed, sharpness, guidance_scale
    ]

    ctrls += [base_model, refiner_model, refiner_switch] + lora_ctrls
    ctrls += [input_image_checkbox, current_tab]
    ctrls += [uov_method, uov_input_image]
    ctrls += [outpaint_selections, inpaint_input_image, inpaint_additional_prompt]
    ctrls += ip_ctrls
    return ctrls


def adps():
    disable_preview=False
    adm_scaler_positive=1.5
    adm_scaler_negative=0.8
    adm_scaler_end=0.3
    adaptive_cfg=modules.config.default_cfg_tsnr
    sampler_name=modules.config.default_sampler
    scheduler_name=modules.config.default_scheduler
    generate_image_grid=False
    overwrite_step=modules.config.default_overwrite_step
    overwrite_switch=modules.config.default_overwrite_switch
    overwrite_width=-1
    overwrite_height=-1
    overwrite_vary_strength=-1
    overwrite_upscale_strength=-1
    mixing_image_prompt_and_vary_upscale=False
    mixing_image_prompt_and_inpaint=False
    debugging_cn_preprocessor=False
    skipping_cn_preprocessor=False
    controlnet_softness=0.25
    canny_low_threshold=64
    canny_high_threshold=128
    refiner_swap_method='joint'
    freeu_enabled =False
    freeu_b1 = 1.01
    freeu_b2 = 1.02
    freeu_s1 = 0.99
    freeu_s2 = 0.95
    freeu_ctrls = [freeu_enabled, freeu_b1, freeu_b2, freeu_s1, freeu_s2]
    debugging_inpaint_preprocessor = False
    inpaint_disable_initial_latent = False
    inpaint_engine = modules.config.default_inpaint_engine_version
    inpaint_strength = 1.0
    inpaint_respective_field = 0.618
    inpaint_ctrls = [debugging_inpaint_preprocessor, inpaint_disable_initial_latent, inpaint_engine, inpaint_strength, inpaint_respective_field]
    adps = [disable_preview, adm_scaler_positive, adm_scaler_negative, adm_scaler_end, adaptive_cfg, sampler_name,
            scheduler_name, generate_image_grid, overwrite_step, overwrite_switch, overwrite_width, overwrite_height,
            overwrite_vary_strength, overwrite_upscale_strength,
            mixing_image_prompt_and_vary_upscale, mixing_image_prompt_and_inpaint,
            debugging_cn_preprocessor, skipping_cn_preprocessor, controlnet_softness,
            canny_low_threshold, canny_high_threshold, refiner_swap_method] 
    adps += freeu_ctrls
    adps += inpaint_ctrls
    return adps


def main():
    advanced_parameters.set_all_advanced_parameters(*adps())
    filePath = r"C:\SoftwareDev\misc_scripts\OpenAIPlayGround\project\tmp\FullResponse1.txt"
    prompts = [i.strip() for i in open(filePath,'r') if i.strip()]
    prompt=""
    image_seed=""
    seed_random = "" is image_seed
    negative_prompt: str=modules.config.default_prompt_negative
    controls=ctrls(prompt,negative_prompt,image_seed, seed_random)
    for p in prompts:
        controls[0] = p
        generate_image(controls.copy())

if __name__=="__main__":
    main()
