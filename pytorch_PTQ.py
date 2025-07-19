import torch
import torch.nn as nn
import torch.quantization
import os
from collections import OrderedDict

from model import Model               # HiNet 전체 구조가 정의된 model.py의 Model
import config as c                    # config.py에서 경로/하이퍼파라미터 등
import modules.Unet_common as common  # DWT 함수 등

torch.backends.quantized.engine = 'fbgemm'

def remove_module_prefix(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict

def run_ptq():
    device = torch.device("cpu")

    # 1. 모델 정의 및 학습 파라미터 로드
    model_fp32 = Model()
    state = torch.load(c.init_model_path, map_location=device)
    state_dict = state['net'] if 'net' in state else state
    state_dict = remove_module_prefix(state_dict)
    model_fp32.load_state_dict(state_dict)
    model_fp32.eval()
    model_fp32.to(device)

    # 2. qconfig 지정
    model_fp32.qconfig = torch.quantization.get_default_qconfig('fbgemm')

    # 3. prepare: observer 삽입
    model_prepared = torch.quantization.prepare(model_fp32, inplace=False)

    # 4. 캘리브레이션
    dwt = common.DWT()
    from torch.utils.data import DataLoader
    from datasets import Hinet_Dataset
    import torchvision.transforms as T
    transform = T.Compose([
        T.CenterCrop(c.cropsize_val),
        T.ToTensor(),
    ])
    val_dataset = Hinet_Dataset(transforms_=transform, mode="val")
    calib_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    with torch.no_grad():
        for i, data in enumerate(calib_loader):
            cover = data['cover_image'].to(device)
            secret = data['secret_image'].to(device)
            cover_input = dwt(cover)
            secret_input = dwt(secret)
            input_img = torch.cat((cover_input, secret_input), 1)
            _ = model_prepared(input_img)
            if i >= 200:
                break

    # 5. convert: INT8 모델로 변환
    int8_model = torch.quantization.convert(model_prepared, inplace=False)

    # 6. **state_dict만 저장!!**
    save_path = os.path.join(c.MODEL_PATH, 'hinet_ptq_int8.pt')
    torch.save(int8_model.state_dict(), save_path)
    print(f"PTQ INT8 state_dict 저장 완료: {save_path}")

if __name__ == "__main__":
    run_ptq()
