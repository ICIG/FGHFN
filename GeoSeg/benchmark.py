import ttach as tta
import multiprocessing.pool as mpp
import multiprocessing as mp
import time
from train_supervision_dp import *
import argparse
from pathlib import Path
import cv2
import numpy as np
import torch

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from datetime import datetime  # 用于记录保存文件的时间

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def label2rgb(mask):
    h, w = mask.shape[0], mask.shape[1]
    mask_rgb = np.zeros(shape=(h, w, 3), dtype=np.uint8)
    mask_convert = mask[np.newaxis, :, :]
    mask_rgb[np.all(mask_convert == 3, axis=0)] = [0, 255, 0]
    mask_rgb[np.all(mask_convert == 0, axis=0)] = [255, 255, 255]
    mask_rgb[np.all(mask_convert == 1, axis=0)] = [255, 0, 0]
    mask_rgb[np.all(mask_convert == 2, axis=0)] = [255, 255, 0]
    mask_rgb[np.all(mask_convert == 4, axis=0)] = [0, 204, 255]
    mask_rgb[np.all(mask_convert == 5, axis=0)] = [0, 0, 255]
    return mask_rgb


def img_writer(inp):
    (mask,  mask_id, rgb) = inp
    if rgb:
        mask_name_tif = mask_id + '.png'
        mask_tif = label2rgb(mask)
        cv2.imwrite(mask_name_tif, mask_tif)
    else:
        mask_png = mask.astype(np.uint8)
        mask_name_png = mask_id + '.png'
        cv2.imwrite(mask_name_png, mask_png)


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-c", "--config_path", type=Path, required=True, help="Path to  config")
    arg("-o", "--output_path", type=Path, help="Path where to save resulting masks.", required=True)
    arg("-t", "--tta", help="Test time augmentation.", default=None, choices=[None, "d4", "lr"])
    arg("--rgb", help="whether output rgb images", action='store_true')
    return parser.parse_args()


def main():
    seed_everything(42)
    args = get_args()
    config = py2cfg(args.config_path)
    args.output_path.mkdir(exist_ok=True, parents=True)
    model = Supervision_Train.load_from_checkpoint(os.path.join(config.weights_path, config.test_weights_name+'.ckpt'), config=config)
    model.cuda(config.gpus[0])
    evaluator = Evaluator(num_class=config.num_classes)
    evaluator.reset()
    model.eval()

    if args.tta == "lr":
        x=1
    elif args.tta == "d4":
        transforms = tta.Compose(
            [
                tta.HorizontalFlip(),
                tta.VerticalFlip(),
                tta.Rotate90(angles=[90,180,270]),
                tta.Scale(scales=[0.5,0.75, 1.0, 1.25, 1.5], interpolation='bicubic', align_corners=False)
            ]
        )
        model = tta.SegmentationTTAWrapper(model, transforms)

    test_dataset = config.test_dataset

    with torch.no_grad():
        total_infer_time = 0.0
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            num_workers=4,
            pin_memory=False,
            drop_last=False,
        )
        results = []
        len_fig = 0
        for input in tqdm(test_loader):
            t0 = time.time()  # 记录开始时间
            # raw_prediction NxCxHxW
            raw_predictions = model(input['img'].cuda(config.gpus[0]))
            t1 = time.time()  # 记录结束时间
            infer_time = t1 - t0  # 计算单张图片的推理时间
            total_infer_time += infer_time  # 累加总推理时间
            len_fig = len_fig + 1
            # image_ids = input["img_id"]
            # masks_true = input['gt_semantic_seg']

            # raw_predictions = nn.Softmax(dim=1)(raw_predictions)
            # predictions = raw_predictions.argmax(dim=1)

            # for i in range(raw_predictions.shape[0]):
            #     mask = predictions[i].cpu().numpy()
            #     evaluator.add_batch(pre_image=mask, gt_image=masks_true[i].cpu().numpy())
            #     mask_name = image_ids[i]
            #     results.append((mask, str(args.output_path / mask_name), args.rgb))
        
        # 计算每张图片平均推理时间
        avg_infer_time = total_infer_time / len_fig
        print('Average inference time per image: {} s'.format(avg_infer_time))
        # 将平均推理时间写入日志文件
        # log_dir = Path(f'./test_log/test_vaihingen_log/{config.test_weights_name}')
        # log_dir.mkdir(parents=True, exist_ok=True)
        # log_filename = log_dir / (datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '.txt')
        # with open(log_filename, 'w') as log_file:
        #     iou_per_class = evaluator.Intersection_over_Union()
        #     f1_per_class = evaluator.F1()
        #     OA = evaluator.OA()
        #     for class_name, class_iou, class_f1 in zip(config.classes, iou_per_class, f1_per_class):
        #         print('F1_{}:{}, IOU_{}:{}'.format(class_name, class_f1, class_name, class_iou))
        #         log_file.write('F1_{}:{}, IOU_{}:{}'.format(class_name, class_f1, class_name, class_iou) + '\n')
        #     print('F1:{}, mIOU:{}, OA:{}'.format(np.nanmean(f1_per_class[:-1]), np.nanmean(iou_per_class[:-1]), OA))
        #     log_file.write('F1:{}, mIOU:{}, OA:{}'.format(np.nanmean(f1_per_class[:-1]), np.nanmean(iou_per_class[:-1]), OA) + '\n')
        #     print('Average inference time per image: {} s'.format(avg_infer_time))
        #     log_file.write('Average inference time per image: {} s'.format(avg_infer_time) + '\n')
        #     t0 = time.time()
        #     mpp.Pool(processes=mp.cpu_count()).map(img_writer, results)
        #     t1 = time.time()
        #     img_write_time = t1 - t0
        #     print('images writing spends: {} s'.format(img_write_time))
        #     log_file.write('images writing spends: {} s'.format(img_write_time) + '\n')

if __name__ == "__main__":
    main()
