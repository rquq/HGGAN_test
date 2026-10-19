# Báo cáo chi tiết: ALPHA, DEV và MAIN

**Phạm vi đối chiếu:** snapshot mã nguồn ngày 05/08/2026  
**Thư mục tương ứng:** `alpha`, `dev`, `main`

## 1. Kết luận điều hành

Ba nhánh không nên được hiểu đơn giản là ba bản Generator khác nhau:

- **ALPHA** là nhánh di sản/tham chiếu. Nó vẫn dùng backbone GBlock cổ điển, nhưng StyleEncoder tạo 32 token không phân vai rõ, fusion xếp chồng Mamba, CNN cục bộ, cross-attention và allographic modulation, còn local discriminator gần như là một bản sao đầy đủ của global discriminator. Training loop dùng toàn bộ cửa sổ 32×32 stride 8, tỷ lệ D:G là 4:1 và một hệ loss cũ có gradient-based dynamic weighting, contextual loss và Gram loss.
- **DEV** là nhánh cải tiến chính. Nó giữ **GBlock backbone cổ điển** để bảo toàn decoder đã chứng minh được khả năng học, nhưng thay gần như toàn bộ đường truyền style–content, cơ chế patch adversarial và cách cân bằng loss. Style code được nén thành **8 token = 1 global + 7 local**; fusion có đúng ba nhiệm vụ: global self-attention cho quan hệ nội dung, depthwise convolution cho liên tục nét, và một allograph router duy nhất cho biến thể ký tự. DEV cũng dùng StrokePatchD nhẹ, crop thích nghi 4–8, global D có width context, R1, gradient clipping, D learning rate riêng và loss weight cố định/dễ diễn giải.
- **MAIN** có mục đích thí nghiệm: lấy hệ thống DEV và thay **GBlock bằng ConditionedRapidBlock lấy cảm hứng từ RapidNet**. Encoder, StyleBackbone, fusion tổng thể, D, StrokePatchD và training loop có cùng thiết kế với DEV. Tuy vậy, snapshot hiện tại chưa phải ablation “chỉ khác G” tuyệt đối: DEV mới có thêm centered allograph routing và RMS cap, trong khi MAIN chưa có; config learning rate, patch weight, routing temperature và decay schedule cũng khác.

Kết luận quan trọng nhất là: **DEV tốt hơn ALPHA không phải vì bỏ local crop**. DEV vẫn dùng local crop, nhưng biến nó thành một tín hiệu có kiểm soát: số crop bị chặn 4–8, được phân tầng theo chiều ngang/dọc, real và fake dùng cùng policy, và patch critic chỉ chịu trách nhiệm về nét chữ cục bộ. Đồng thời DEV loại bỏ việc style bị tiêm lặp lại qua quá nhiều tầng fusion và loại bỏ hệ dynamic loss weighting vốn so sánh gradient ở các không gian không tương đồng.

## 2. Bảng tổng quan

| Thành phần | ALPHA | DEV | MAIN |
|---|---|---|---|
| Vai trò | Nhánh tham chiếu/di sản local crop | Nhánh nghiên cứu chính | Thí nghiệm đổi G của DEV |
| G backbone | 4 × classic GBlock | 4 × classic GBlock | 4 × ConditionedRapidBlock |
| Upsampling | Interpolate mặc định + GBlock | Interpolate mặc định + GBlock | Bilinear + depthwise spatial transition |
| Style token | 32 token ngang vai | 8 token: 1 global + 7 local | Như DEV |
| Fusion | Bi-Mamba → local CNN → cross-attention → allograph modulation → global affine | Global style-conditioned self-attention → local depthwise → allograph modulation | Cùng ba-stage fusion nền tảng của DEV |
| Global D | BigGAN D + global pooling | BigGAN D + low-resolution width context | Như DEV |
| Local D | Một BigGAN D thứ hai | StrokePatchD chuyên biệt nét chữ | Như DEV |
| Patch policy thực tế | Toàn bộ patch 32×32, stride 8 | 4–8 crop thích nghi/ảnh | Như DEV |
| D:G update | 4:1 | 2:1 | 2:1 |
| Loss balancing | Dynamic gradient ratios + Gram + contextual | Weight rõ ràng, cố định/có schedule | Như DEV |
| EMA | G, E và cả frozen B | Chỉ G và E | Chỉ G và E |
| Validation logging | Chủ yếu theo iteration | Theo epoch + CSV trong run/checkpoint | Như DEV |

## 3. Kích thước mô hình thực tế

Số dưới đây là trainable parameters khi khởi tạo trực tiếp từ `configs/gan_iam.yml` của từng nhánh:

| Module | ALPHA | DEV | MAIN |
|---|---:|---:|---:|
| Generator G | 4,339,329 | 3,984,897 | 2,375,169 |
| StyleEncoder E | 2,276,481 | 2,272,977 | 2,272,977 |
| Global D | 2,366,145 | 2,628,802 | 2,628,802 |
| Patch D | 2,366,145 | 411,777 | 411,777 |
| Riêng fusion | 661,184 | 306,760 | 306,760 |
| Tổng G+E+D+P | 11,348,100 | 9,298,453 | 7,688,725 |

So với ALPHA:

- Fusion của DEV giảm khoảng **53.6% parameters**.
- G+E của DEV giảm khoảng **5.4%**.
- Cặp D+P của DEV giảm khoảng **35.7%**, dù global D mạnh hơn do có width-context mixer.
- Tổng bốn module trainable chính giảm khoảng **18.1%**.

MAIN tiếp tục giảm G khoảng **40.4% so với DEV**. Tuy nhiên, ít parameters không đồng nghĩa chắc chắn chạy nhanh hơn: nhiều depthwise/dilated kernel nhỏ có thể memory-bound và tạo nhiều kernel launch trên T4, còn bilinear interpolation cũng đắt hơn nearest interpolation mặc định.

## 4. Luồng của ALPHA

```text
Ảnh style/crop
  └─ StyleBackbone + multi-scale features
       └─ 32 learned queries cross-attend toàn bộ spatial features
            └─ 32 style tokens, không có global/local contract rõ ràng

Text IDs
  └─ character embedding
       └─ bidirectional Style-Cross Mamba
            └─ gated local depthwise CNN
                 └─ character-conditioned style cross-attention
                      └─ character-conditioned allographic modulation
                           └─ global affine từ trung bình 32 token
                                └─ classic GBlocks → ảnh

Ảnh real/fake
  ├─ Global BigGAN D
  └─ tất cả patch 32×32 stride 8 → BigGAN Patch D
```

Điểm cốt lõi: style được đưa vào content nhiều lần liên tiếp. Mamba đã cross-condition bằng style, sau đó cross-attention lại query style, allograph modulation lại query style, và cuối cùng toàn bộ 32 token lại được average để scale/shift global. Các stage đều hợp lý nếu đứng riêng, nhưng khi xếp chồng thì ranh giới nhiệm vụ biến mất.

Hậu quả có khả năng cao:

1. Một stage sửa representation mà stage sau vừa học xong cách sửa lại.
2. Gradient từ nhiều đường style cạnh tranh trên cùng token bank.
3. Local allograph và global writer identity không được tách thành hai kênh rõ ràng.
4. Scale/shift cuối đường truyền không bị giới hạn, nên style artifact có thể được khuếch đại trước G.
5. 32 token làm attention/cross-routing đắt hơn nhưng không đảm bảo 32 chuyên môn khác nhau; token collapse vẫn có thể xảy ra.

Đây là lời giải thích kiến trúc hợp lý nhất cho quan sát “style của ALPHA trông sai một cách kỳ lạ”: không nhất thiết StyleEncoder thiếu capacity; ngược lại, **style có quá nhiều đường can thiệp và thiếu contract về token nào chịu trách nhiệm cho thông tin nào**.

## 5. Luồng của DEV

```text
Ảnh gốc làm style reference
  └─ frozen StyleBackbone, mask đúng valid width ở mọi scale
       ├─ masked global pooling → token 0: global writer style
       └─ 7 orthogonal local queries + fixed slot anchors
            └─ token 1..7: local stroke/allograph style

Text IDs
  └─ character embedding
       ├─ Stage 1: global-style-conditioned self-attention
       ├─ Stage 2: gated depthwise local continuity
       └─ Stage 3: character-conditioned routing tới 7 local tokens
            └─ bounded residual allographic modulation
                 └─ outer fusion residual gate
                      └─ classic GBlocks → ảnh

Ảnh real/fake
  ├─ Global D + width context
  └─ matched adaptive crops → StrokePatchD
```

DEV tổ chức lại fusion theo nguyên tắc **mỗi stage có một nhiệm vụ duy nhất**:

### Stage 1 — quan hệ content toàn từ

Self-attention chạy giữa các character token, không dùng local style làm key/value. Token global chỉ điều kiện hóa normalization và residual strength. Stage này học spacing, liên hệ ký tự, nhịp từ và bố cục toàn cục theo writer style.

Điểm quan trọng là nó không cạnh tranh với allograph router. Global style nói “toàn từ nên mang nhịp và độ nghiêng như thế nào”, chứ không nói “ký tự `a` phải chọn local slot nào”.

### Stage 2 — liên tục nét cục bộ

Depthwise Conv1d kernel 5 xử lý neighborhood theo trục ký tự. Một pointwise projection và selector gate quyết định kênh nào cần hiệu chỉnh. Residual gate khởi tạo 0.25 nên nhánh này có gradient ngay từ đầu nhưng không ghi đè content embedding.

Nó bổ sung inductive bias cục bộ mà self-attention không đảm bảo: nối nét, khoảng cách hai ký tự cạnh nhau, ascender/descender transition và độ liên tục đường bút.

### Stage 3 — allograph duy nhất

Đây là content-to-style attention duy nhất trong fusion. Mỗi character token query 7 local style slot. Router dùng cả content context và character ID, nên `a`, `b`, `r`, `s`, `h`, `d` có thể chọn/phối local evidence khác nhau.

Snapshot DEV mới nhất còn có hai safeguard:

- **Centered routing keys:** trước khi tạo key, trung bình của 7 local slot được trừ khỏi từng slot. Phần writer identity chung vẫn được giữ trong value. Router vì vậy tập trung vào khác biệt allograph giữa các slot, thay vì tất cả slot cùng bị một writer offset kéo theo một hướng.
- **RMS cap = 1.0 trước tanh:** scale/shift quá lớn được nén trước khi vào bounded tanh, tránh toàn bộ kênh rơi vào vùng bão hòa và mất gradient.

Sau đó:

- cosine query/key loại bỏ ảnh hưởng magnitude;
- learned temperature bị chặn để router không sắc vô hạn;
- character gain bị giới hạn 0.25;
- scale và shift bị giới hạn ±0.3;
- modulation đi qua residual gate khởi tạo 0.5;
- outer fusion gate trong G khởi tạo 0.25, nên raw content path vẫn tồn tại.

Vì vậy encoder → fusion → G có nhiều đường residual liên tục, nhưng không có nhiều module cùng làm một việc. Đây là khác biệt thiết kế quan trọng nhất so với ALPHA.

## 6. StyleEncoder: 32 token cũ và 8 token mới

### ALPHA

- Số style token bị gắn cứng bằng `style_dim`, tức 32.
- Cả 32 learned query có cùng vai trò.
- Không có token global được reserve.
- Không có orthogonal initialization hay fixed slot identity.
- Không truyền `key_padding_mask` vào multi-head attention.
- Feature vùng padding có thể tham gia pooling/attention.
- Projection layer có logic có thể rebuild trong `forward`; nếu backbone channels thay đổi, parameter mới có thể được tạo sau optimizer và không được optimize đúng cách.
- Không có adversarial content probe để ép style code bỏ thông tin chữ.

### DEV/MAIN

- `num_style_tokens=8` được tách khỏi `style_dim=32`.
- Token 0 lấy từ masked global pooling và có contract rõ: writer-level/global style.
- 7 local query được khởi tạo trực giao, giảm khả năng collapse ngay đầu training.
- Fixed orthogonal slot anchors cho mỗi local slot một identity bền vững.
- Local attention có gated residual khởi tạo 0.25 thay vì thay thế query hoàn toàn.
- Affine-free LayerNorm cân bằng scale giữa feature maps của frozen backbone.
- Mask valid width được áp dụng ở main feature, mọi multi-scale feature, spatial token và attention padding mask.
- Tất cả projection được xây trong `__init__`, trước khi optimizer được tạo.
- Content probe với gradient reversal phạt style code nếu nó còn dự đoán được nội dung ký tự.

Việc giảm 32 xuống 8 token không phải cắt capacity tùy tiện. Mỗi token vẫn có vector 32 chiều, nhưng số slot được chọn theo số vai trò thực tế. Bảy local slot đủ để hình thành một vocabulary mềm của stroke/allograph trong word-level synthesis; quan trọng hơn, chúng có cơ chế chống collapse và được fusion sử dụng đúng vai trò.

## 7. Patch/crop: khác biệt thật sự giữa ALPHA và DEV

Tên ALPHA không phản ánh hoàn toàn code hiện tại. Trong training loop của snapshot này, hàm được gọi là `extract_all_patches`, không phải `PatchSampler.random_sample`:

- patch size: 32×32;
- stride ngang/dọc: 8;
- lấy toàn bộ patch hợp lệ;
- số patch tăng gần tuyến tính theo chiều rộng từ và có hai hàng theo chiều cao 64.

Đây là exhaustive sliding-window supervision. Nó cho độ phủ dày nhưng có ba nhược điểm:

1. Các patch lân cận overlap 75%, tạo rất nhiều mẫu gần trùng nhau.
2. Từ dài tạo nhiều patch hơn và chi phối local discriminator/loss.
3. Patch D là một BigGAN discriminator đầy đủ, nên chi phí D local rất lớn.

DEV dùng `sample_adaptive_patches`:

- patch size vẫn là 32×32;
- số patch = `ceil(valid_width/32)`, chặn trong [4, 8];
- trục ngang được chia thành các strata để tránh lấy lặp một vùng;
- trục dọc luân phiên nửa trên/nửa dưới để thấy ascender, baseline và descender;
- real, augmented-real, random fake, style-transfer fake và reconstruction fake đều dùng cùng policy;
- patch chỉ vào StrokePatchD, không được giả làm “ảnh từ hoàn chỉnh” trước global D.

Nói ngắn gọn: ALPHA ưu tiên **mật độ crop**, DEV ưu tiên **độ phủ cân bằng và đúng nhiệm vụ**.

## 8. Discriminator

### ALPHA

Global D và Patch D đều là BigGAN discriminator khoảng 2.366M parameters. Patch D chỉ kế thừa nguyên global D với resolution 32; nó không có block chuyên biệt cho stroke. D local vì thế học được realism cục bộ, nhưng dùng quá nhiều capacity cho một nhiệm vụ nhỏ và không có inductive bias riêng cho nét ngang/dọc.

Global D sum-pool feature theo không gian sau convolution. Nó biết valid width, nhưng không có một stage tương tác toàn chiều rộng sau khi rút gọn feature.

### DEV/MAIN

Global D thêm một width-context mixer ở feature map thấp:

- gộp chiều cao nhưng giữ chuỗi chiều rộng;
- manual multi-head attention trên phần width hợp lệ;
- residual scale nhỏ;
- sau đó mới masked sum pooling.

Mục tiêu là phát hiện coherence của toàn từ: spacing bất thường, stroke dư kéo dài, baseline đứt, đoạn ký tự không ăn nhập. Manual attention được dùng thay fused efficient SDPA để đảm bảo backward hoạt động trên các tổ hợp PyTorch/CUDA cũ.

StrokePatchD chỉ có khoảng 0.412M parameters. Mỗi block dùng:

- convolution đầu vào;
- depthwise 1×5 cho nét ngang;
- depthwise 5×1 cho nét dọc;
- pointwise fusion;
- residual shortcut và average pooling.

Đây là critic phù hợp hơn với handwriting local structure, nhẹ hơn Patch D cũ khoảng 82.6%, và xuất spatial logits thay vì ép mọi patch vào một global scalar quá sớm.

DEV còn dùng lazy R1 mỗi 16 iteration với `lambda_r1=0.01`, giúp global D bớt tạo decision boundary sắc/quá khớp. Patch adversarial contribution được weight 0.75 trong DEV thay vì mặc định cộng nguyên 1.0 như ALPHA.

## 9. Loss functions

### 9.1. ALPHA

Loss G cũ về bản chất là:

```text
L_G = L_adv_global + L_adv_patch
    + gp_ctc  · (CTC_random + CTC_transfer + CTC_reconstruction)
    + gp_info · (style_reconstruction + style_contrastive)
    + gp_wid  · writer_identification
    + gp_recn · pixel_reconstruction
    + 2.0 · contextual
    + 2.0 · Gram
    + 0.0001 · KL
```

`gp_*` không phải gradient penalty của D. Chúng là dynamic weight lấy tỷ lệ standard deviation của nhiều gradient. Vấn đề là các gradient được đo ở các không gian khác nhau:

- adversarial gradient theo image tensor;
- OCR gradient theo OCR logits;
- writer gradient theo writer logits;
- reconstruction gradient theo latent style code.

Do không cùng đơn vị/shape/semantic space, tỷ lệ standard deviation của chúng không phải một thước đo cân bằng objective đáng tin cậy. Weight có thể dao động do topology của head, batch composition hay saturation, chứ không chỉ do loss nào đang yếu/mạnh.

Gram và contextual cũng cùng hướng tới feature/style similarity. Với log từng quan sát trước đây, raw Gram có thể lên hàng trăm/hàng nghìn; nhân 2.0 làm nó có khả năng áp đảo tổng objective về mặt số học. Style reconstruction và style contrastive lại tiếp tục giám sát cùng cặp latent. Hệ thống vì thế có nhiều mục tiêu hợp lệ nhưng chồng lấn và khó biết gradient nào đang điều khiển G.

### 9.2. DEV/MAIN

Loss hiện tại được gom theo vai trò:

```text
L_adv     = L_global + λ_patch · L_patch
L_content = λ_ctc(epoch) · (CTC_random + CTC_transfer)
L_recn    = 10.0 · L1_reconstruction
L_style   = 1.0 · style_info
          + 1.0 · style_transfer_cycle
          + 0.02 · content_adversary
L_writer  = 0.5 · writer_identification
L_context = 0.1 · masked_contextual
L_KL      = 0.1 · KL
L_G       = tổng các nhóm trên
```

Các thay đổi quan trọng:

- Không còn Gram loss.
- Không còn style contrastive loss cũ.
- Không còn dynamic `gp_*` weighting.
- CTC reconstruction bị bỏ vì reconstruction đã có exact pixel target; CTC chỉ giữ cho random generation và style transfer.
- CTC giảm tuyến tính từ 1.0 xuống 0.6 trong epoch 10–20, để đầu training giữ content rồi dần nhường capacity cho style.
- Contextual loss chỉ so style-transfer image với style reference, có width mask.
- Writer loss chỉ đánh style-transfer image, không lặp trên reconstruction.
- Log ghi **weighted contribution**, nên số trên W&B phản ánh đúng phần loss thực sự đi vào tổng G.
- G/E được gradient clip ở norm 5.0.

Các loss DEV còn lại không hoàn toàn redundant:

- `style_info` buộc random latent sinh ra có thể encode trở lại.
- `style_cycle` buộc style lấy từ ảnh reference được giữ sau transfer.
- `writer_id` bảo toàn identity ở mức class writer.
- `contextual` bảo toàn feature style không cần pixel alignment.
- `content_adv` loại content leakage khỏi style code.
- `KL` định hình phân phối latent cho sampling.

Có overlap nhẹ giữa writer/contextual và giữa KL/content-adversary, nhưng chúng ràng buộc các failure mode khác nhau. Nếu cần ablation, nên tắt từng regularizer nhỏ (`content_adv`, rồi `contextual`) ở các run riêng; không nên xóa đồng thời các loss cốt lõi. So với ALPHA, DEV đã là hệ loss gọn và dễ kiểm soát hơn đáng kể.

## 10. Optimization và schedule

| Config | ALPHA | DEV | MAIN |
|---|---:|---:|---:|
| Epoch target | 70 | 50 | 50 |
| Batch size | 6 | 8 | 8 |
| Eval batch | 6 | 12 | 12 |
| G/E LR | 2e-4 | 1.5e-4 | 2e-4 |
| D/P LR | 2e-4 | 5e-5 | 1e-4 |
| D:G | 4:1 | 2:1 | 2:1 |
| Start decay | 25 | 10 | 12 |
| Decay length | 46 | 20 | 20 |
| LR floor | gần 0.001× theo implementation cũ | 0.1× | 0.1× |
| Patch weight | 1.0 ngầm định | 0.75 | 0.5 |

ALPHA đặt cùng LR cho G và D, đồng thời cập nhật D bốn lần cho mỗi lần cập nhật G. Với hai discriminator lớn, đây là cấu hình rất dễ làm critic đi trước generator. DEV giảm cả update ratio lẫn D learning rate, nhưng vẫn giữ hai critic step để local/global D không bị tụt.

Một epoch không tương đương giữa hai nhánh: batch size và D:G khác nhau. Khi so sánh nên dùng số ảnh đã thấy, số lần G update, wall-clock và cùng validation corpus; không chỉ nhìn epoch.

## 11. MAIN — thí nghiệm thay G

Mục đích khoa học của MAIN là trả lời câu hỏi:

> Khi giữ encoder, fusion và adversarial supervision của DEV, một decoder RapidNet-inspired nhẹ hơn có tốt hơn classic GBlock hay không?

### DEV GBlock

Mỗi stage dùng conditional BatchNorm, activation, hai spectral-normalized spatial convolution và residual shortcut. Upsampling mặc định của `F.interpolate` là nearest. Đây là đường sinh ảnh có capacity cao và đã ổn định qua nhiều run.

### MAIN ConditionedRapidBlock

Mỗi stage dùng:

1. conditional BN + activation;
2. bilinear upsampling;
3. pointwise projection;
4. depthwise 3×3 spatial transition để phá các cell lặp sau upsample;
5. conditional BN;
6. RapidMLDC mixer gồm depthwise 7×7 positional path, dilated depthwise 3×3 với dilation 2 và 3, pointwise mixing;
7. large-kernel depthwise FFN và pointwise expansion ratio 2;
8. residual scale 0.1 cho mixer/FFN;
9. pointwise shortcut.

Ưu điểm kỳ vọng:

- receptive field lớn với ít parameters;
- multi-scale stroke context;
- bilinear upsampling + depthwise transition giảm blocky replication;
- G nhẹ hơn khoảng 40.4%.

Rủi ro:

- depthwise/dilated/large-kernel path có thể chậm thực tế trên T4 dù ít FLOPs/parameters;
- bốn stage đều bị thay một lúc, nên đây là perturbation lớn chứ không phải thay một block nhỏ;
- dilated branches dễ tạo secondary stroke/echo nếu critic hoặc loss khuyến khích high-frequency detail quá mạnh;
- capacity cross-channel thấp hơn GBlock, phụ thuộc nhiều vào pointwise layers;
- residual MLDC có thể dần tăng cường các nét nhỏ dư ở epoch muộn.

Quan sát trước đó của MAIN — nét nhỏ dư, line phụ và artifact tăng ở epoch muộn — phù hợp với giả thuyết các multi-dilation path đang over-refine, nhưng đây vẫn là **suy luận**, không phải bằng chứng nhân quả nếu config và fusion chưa được giữ hoàn toàn giống DEV.

### Caveat của snapshot hiện tại

Các file `module.py`, `model.py`, `utils.py` và `masking.py` của DEV và MAIN hiện giống hệt byte-for-byte; D và PatchD cũng cùng code. Tuy nhiên còn các khác biệt ngoài G:

- DEV có centered routing keys; MAIN chưa có.
- DEV có modulation RMS cap; MAIN chưa có.
- DEV routing temperature 0.5; MAIN 0.7. Nhiệt độ 0.5 khởi tạo routing sắc hơn.
- DEV dùng G LR 1.5e-4, D LR 5e-5, patch weight 0.75, decay bắt đầu epoch 10.
- MAIN dùng G LR 2e-4, D LR 1e-4, patch weight 0.5, decay bắt đầu epoch 12.

Do đó cách diễn đạt đúng là: **MAIN có mục đích là experiment thay G, nhưng trạng thái code/config hiện tại có thêm một số delta nhỏ ngoài G**. Muốn benchmark sạch, cần đóng băng cùng một fusion snapshot và cùng config cho cả hai; chỉ cho phép `GBlock ↔ ConditionedRapidBlock` khác nhau.

## 12. Tại sao DEV có cơ sở tốt hơn ALPHA

DEV giải quyết đồng thời bốn failure mode của ALPHA:

1. **Style token collapse/content leakage:** tách 1 global + 7 local, orthogonal query, slot anchor, mask padding và gradient reversal.
2. **Fusion redundancy:** thay chuỗi Mamba + cross-attention + modulation + global affine bằng ba stage có contract riêng.
3. **Local critic quá nặng và crop thiên lệch:** thay exhaustive overlapping patches + full BigGAN P bằng crop phân tầng 4–8 + StrokePatchD.
4. **Loss competition:** bỏ Gram/contrastive/dynamic gradient ratio; dùng weight có ý nghĩa và log weighted contribution.

Điều này cũng giải thích vì sao DEV có thể học `r`, `s`, descender của `y/p/g` tốt hơn baseline/ALPHA: allograph evidence được dành riêng cho local slots và character router, trong khi global style/context không còn liên tục ghi đè local choice.

## 13. Cách nên sử dụng ba nhánh

- **ALPHA:** giữ làm historical reference và để ablate hiệu ứng của hệ fusion/loss/crop cũ. Không nên xem nó là model mục tiêu để tiếp tục mở rộng bằng cách thêm loss hoặc thêm token.
- **DEV:** dùng làm canonical model cho nghiên cứu style encoder, fusion, allograph và training recipe. Đây là nhánh có flow dễ giải thích và ổn định nhất về mặt thiết kế.
- **MAIN:** dùng làm generator ablation. Mọi thay đổi ngoài G nên được mirror từ DEV trước khi chạy controlled comparison, hoặc phải ghi rõ trong experiment log.

Đối với báo cáo kết quả, nên ghi tối thiểu:

- commit/snapshot của fusion;
- G block type;
- G/D LR;
- D:G ratio;
- patch weight và crop range;
- số G updates;
- FID/KID/HWD/CMMD ở cùng epoch-equivalent;
- checkpoint thường hay EMA được dùng để eval.

## 14. Kết luận cuối

Khác biệt lớn nhất giữa DEV và ALPHA không nằm ở việc “có crop hay không”. Cả hai đều có local patch supervision. Khác biệt nằm ở **phân công trách nhiệm**:

- ALPHA dùng nhiều style token, nhiều fusion layer và nhiều loss cùng lúc, nhưng thiếu ranh giới rõ giữa global writer style, local stroke style và character-specific allograph.
- DEV giảm số đường can thiệp, tách global/local style, chỉ để một router làm allograph, giới hạn cường độ modulation, giữ residual content path, và dùng local critic/crop đúng chuyên môn.

MAIN sau đó là bước thử nghiệm hợp lý: giữ hệ thống DEV và hỏi liệu G nhẹ kiểu RapidNet có thể thay classic GBlock hay không. Nhưng kết quả MAIN chỉ có thể quy hoàn toàn cho G khi fusion và config được đồng bộ tuyệt đối với DEV. Ở snapshot hiện tại, DEV vẫn là reference kiến trúc sạch hơn; MAIN là thí nghiệm G đầy tiềm năng nhưng chưa chứng minh được rằng mức giảm 40% parameters chuyển thành chất lượng hoặc tốc độ tốt hơn.
