import torch
from model import Model
import modules.Unet_common as common

torch.backends.quantized.engine = 'fbgemm'
print("Current engine:", torch.backends.quantized.engine)

def run_int8_infer():
    device = torch.device("cpu")

    # 1. 모델 생성 및 qconfig 지정 (반드시 PTQ 저장 때와 동일하게)
    model_fp32 = Model()
    model_fp32.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    
    # 2. convert (구조만 int8로 바꿈, 파라미터는 아직 X)
    model_int8 = torch.quantization.convert(model_fp32.eval(), inplace=False)
    model_int8.to(device)

    # 3. INT8 state_dict만 로드 (파라미터 로딩)
    state_dict = torch.load("/root/InvisMark/model/hinet_ptq_int8.pt", map_location="cpu")
    model_int8.load_state_dict(state_dict, strict=False)
    model_int8.eval()

    # 4. 입력 생성 및 추론
    cover = torch.rand(1, 3, 256, 256)
    secret = torch.rand(1, 3, 256, 256)
    dwt = common.DWT()
    cover_dwt = dwt(cover)
    secret_dwt = dwt(secret)
    input_img = torch.cat((cover_dwt, secret_dwt), dim=1)

    with torch.no_grad():
        output = model_int8(input_img)
    print("INT8 추론 출력 shape:", output.shape)

if __name__ == "__main__":
    run_int8_infer()
