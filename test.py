import os
from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import Visualizer
from util import html
import numpy as np
import cv2

if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip

    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    model = create_model(opt)
    visualizer = Visualizer(opt)

    # Create results webpage
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

    # Test loop
    for i, data in enumerate(dataset):
        if i >= opt.how_many:
            break
        
        model.set_input(data)

        img_path = model.get_image_paths()
        print('%04d: process image... %s' % (i, img_path))  # Ensuring print order
        
        model.test()  # Run model inference
        # Generate RDEB heatmap from first encoder
        # rdeb_heatmap = model.get_edgeproj_heatmap('encoder3_edge')  # [H,W]

        # Generate saliency map (requires GT if available)
        # saliency_map = model.compute_saliency_map(use_gt=True)  # [H,W]

        # Add to visuals dict for saving


        if opt.dataset_mode == 'aligned_mat':
            visuals = model.get_current_visuals()
            img_path[0] = img_path[0] + str(i)
        elif opt.dataset_mode == 'unaligned_mat':   
            visuals = model.get_current_visuals()
            slice_select = [opt.input_nc // 2] * 3  # Ensure integer index
            visuals['real_A'] = visuals['real_A'][:, :, slice_select]
            visuals['real_B'] = visuals['real_B'][:, :, slice_select]
            visuals['fake_A'] = visuals['fake_A'][:, :, slice_select]
            visuals['fake_B'] = visuals['fake_B'][:, :, slice_select]
            visuals['rec_A'] = visuals['rec_A'][:, :, slice_select]
            visuals['rec_B'] = visuals['rec_B'][:, :, slice_select]
            img_path[0] = img_path[0] + str(i)
        else:
            visuals = model.get_current_visuals()
        # # Scale to [0,255] uint8
        # rdeb_gray = (rdeb_heatmap * 255).astype(np.uint8)   # [H,W]
        # sal_gray  = (saliency_map * 255).astype(np.uint8)   # [H,W]

        # # Apply colormap (JET is classic for heatmaps)
        # rdeb_color = cv2.applyColorMap(rdeb_gray, cv2.COLORMAP_JET)   # [H,W,3], BGR
        # sal_color  = cv2.applyColorMap(sal_gray,  cv2.COLORMAP_JET)   # [H,W,3], BGR

        # # Convert BGR -> RGB because your visualizer expects RGB
        # rdeb_color = cv2.cvtColor(rdeb_color, cv2.COLOR_BGR2RGB)
        # sal_color  = cv2.cvtColor(sal_color,  cv2.COLOR_BGR2RGB)

        # visuals['rdeb_heatmap'] = rdeb_color
        # visuals['saliency_map'] = sal_color
            # 3) encoder stage heatmaps (same stages for baseline and RDEB)
        # enc1_heat = model.get_encoder_heatmap('encoder_1')  # [H,W]
        # enc2_heat = model.get_encoder_heatmap('encoder_2')  # [H,W]
        # enc3_heat = model.get_encoder_heatmap('encoder_3')  # [H,W]

        # 4) saliency map (if you have compute_saliency_map in your train model; 
        # if not, you can skip this line or create a test-specific version)
        saliency_map = model.compute_saliency_map(use_gt=True)  # [H,W] (if available)

        # ===== convert [H,W] gray -> color maps =====
        # def to_color(hmap):
        #     gray = (hmap * 255).astype(np.uint8)
        #     color = cv2.applyColorMap(gray, cv2.COLORMAP_JET)       # BGR
        #     color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)          # RGB
        #     return color

        # enc1_color = to_color(enc1_heat)
        # enc2_color = to_color(enc2_heat)
        # enc3_color = to_color(enc3_heat)
        # sal_color  = to_color(saliency_map) if saliency_map is not None else None

        # visuals['enc1_heatmap'] = enc1_color
        # visuals['enc2_heatmap'] = enc2_color
        # visuals['enc3_heatmap'] = enc3_color
        # if sal_color is not None:
        #     visuals['saliency_map'] = sal_color

        # # 5) optional: overlay on FLAIR (assuming visuals['real_A'] is RGB T1/T2/FLAIR)
        # if 'real_A' in visuals:
        #     base = visuals['real_A'].copy()  # H×W×3 uint8
        #     alpha = 0.5
        #     visuals['enc1_overlay'] = cv2.addWeighted(base, 1 - alpha, enc1_color, alpha, 0)
        #     visuals['enc2_overlay'] = cv2.addWeighted(base, 1 - alpha, enc2_color, alpha, 0)
        #     visuals['enc3_overlay'] = cv2.addWeighted(base, 1 - alpha, enc3_color, alpha, 0)
        #     if sal_color is not None:
        #         visuals['saliency_overlay'] = cv2.addWeighted(base, 1 - alpha, sal_color, alpha, 0)

    # 6) save everything via your existing visualizer
        visualizer.save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio)

    # Compute final PSNR & SSIM after all images are processed
    model.compute_final_metrics()
    
    webpage.save()
