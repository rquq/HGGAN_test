# Báo cáo cải tiến DEV: từ khi thêm texture generator đến 10/09/2026

Ngày lập: 10/09/2026. Nhánh được kiểm tra: `dev`, repository `/home/quq/machineLearning/HTG/HGGAN_test/dev`. HEAD trước khi viết báo cáo: `475302107b7fd5ab2d986985fb5a67bb4cd0c647`.

**Cập nhật phạm vi theo yêu cầu:** báo cáo đã mở rộng về commit `009014e` lúc **23:02:24 ngày 30/08/2026**, là lần đầu thêm `networks/texture_generator.py` vào DEV. Toàn khoảng mở rộng có **16 commit**. Phần 1–13 giữ phân tích chi tiết của giai đoạn 01–10/09; phần 14–17 bổ sung đầy đủ chuỗi thay đổi từ 30/08, nguồn gốc texture module và đánh giá nó có cần thiết không. Giữ nguyên tên file để các link đã gửi vẫn dùng được.

## 1. Phạm vi và kết quả chính

Báo cáo lấy **10 ngày lịch gần nhất, từ 01/09 đến hết 10/09/2026, múi giờ UTC+7**. Git local có 5 commit trong khoảng này; commit mới nhất là ngày 06/09. Tất cả có cùng tiêu đề “Update model configuration and training runs”, vì vậy nội dung dưới đây được xác định bằng diff thực tế, không suy đoán từ commit message. Một số thay đổi nền ngày 30–31/08 được ghi riêng vì chúng giải thích nguồn gốc của cơ chế chống lỗi khi lấy `/`, `.`, `,` làm style reference.

Trọng tâm của giai đoạn này là làm cho thông tin style ít ỏi được sử dụng có kiểm soát, sửa cách đưa ký tự hiếm vào training, giảm nhiễu trong character conditioning của StrokePatchD, và làm phép đo KID đáng tin cậy hơn. Đồng thời, code giảm các lượt chạy backbone và đồng bộ CPU–GPU dư thừa.

Cần phân biệt hai bài toán:

- **Sinh một từ dài từ ảnh reference chỉ có `/`, `.`, `,`…:** ảnh reference không đủ bằng chứng về toàn bộ phong cách viết. Hướng xử lý nằm chủ yếu ở StyleEncoder và cách kiểm soát local style.
- **Sinh đúng chính các ký tự hiếm trong câu/từ đích:** phụ thuộc mức độ xuất hiện của chúng trong training text, OCR supervision và local discriminator. Đây là lý do thay đổi rare-word sampling và patch conditioning có ý nghĩa.

Các commit chứng minh những thay đổi đã được triển khai. Chúng **chưa tự chứng minh KID giảm bao nhiêu, FID đạt dưới 4, hoặc lỗi punctuation đã hết**. Phiên này kiểm tra lịch sử và code; không chạy lại GAN hay một ablation dài để gán mức cải thiện cho từng thành phần.

## 2. Bảng lịch sử commit trong kỳ

| Thời gian UTC+7 | Commit | Nội dung xác nhận bằng diff | Giá trị đáng báo cáo |
|---|---|---|---|
| 01/09 01:08 | [597d65b](https://github.com/rquq/HGGAN_test/commit/597d65b3edaa5f9441304107879ae0063c8b39d5) | Bổ sung checkpoint OCR32 và W32; xóa `style_prior_mean.npy` | Cập nhật tài nguyên pretrained x32; không phải bằng chứng một kiến trúc mới |
| 06/09 02:03 | [84dec47](https://github.com/rquq/HGGAN_test/commit/84dec47dfff23dd3579e7e85693bad3adc0fcf1b) | Sửa length scale x32; sửa support factor cho reference ngắn; đưa local style đã ổn định vào nhánh VAE | Thay đổi trực tiếp nhất cho vấn đề reference `/`, `.`, `,` |
| 06/09 02:59 | [1f48857](https://github.com/rquq/HGGAN_test/commit/1f48857e8f811316a308bd1a23ba6c4ab26fd5e7) | Startup summary gọn, RESULTS table, trạng thái job, theo dõi epoch và metric cuối | Cải thiện vận hành và khả năng đọc kết quả GAN/OCR/W |
| 06/09 10:53 | [91b107d](https://github.com/rquq/HGGAN_test/commit/91b107d980d90f2b0b69e61e04ec9d10b07b0792) | Thay weights OCR32 và OCR64 | Cập nhật teacher R sau pretraining; cần eval riêng để kết luận tốt hơn |
| 06/09 15:46 | [4753021](https://github.com/rquq/HGGAN_test/commit/475302107b7fd5ab2d986985fb5a67bb4cd0c647) | Rare-word sampling, patch confidence, vector hóa crop, sửa config encoder, tái sử dụng backbone feature, cache float32 cho metrics, cập nhật W32/W64 | Gói thay đổi lớn nhất về phân phối training, độ đúng của code, tốc độ và đánh giá KID |

Link commit dùng remote đang cấu hình trong repository. Các dẫn chứng về nội dung được kiểm tra từ Git local.

## 3. Sinh từ reference chỉ có một ký tự đặc biệt

### 3.1. Vì sao reference `/` hoặc `.` dễ gây lỗi?

StyleEncoder hiện có 8 style tokens: một token global và bảy token local. Local queries thu thập thông tin nét từ nhiều feature maps của backbone, sau đó cung cấp style cho fusion và Generator.

Với reference là một từ đủ dài, encoder quan sát nhiều nét, cách nối chữ, độ nghiêng và biến thể hình dạng. Một dấu chấm hoặc dấu gạch chéo chỉ cung cấp một phần rất nhỏ trong số đó. Nếu các local slots vẫn tạo ra biến thiên mạnh, Generator có thể áp biến thiên ấy lên toàn bộ từ đích: nét bị kéo dài, độ dày/độ nghiêng bị phóng đại, hoặc cấu trúc từ trở nên không tự nhiên. Đây là cách giải thích theo cơ chế code; xác nhận nguyên nhân cho một ảnh cụ thể vẫn cần chạy đối chứng cùng checkpoint.

### 3.2. Nền trước kỳ báo cáo: 30/08

[009014e](https://github.com/rquq/HGGAN_test/commit/009014e4e67c147710c65d12db6ad0a137b8efc6) đã thêm hai cơ chế:

1. Giảm chênh lệch giữa local style và global style khi feature sequence quá ngắn, bằng `clamp(feature_length / 4, 0.25, 1)`.
2. Chặn L2 norm của vector mean style tại `9.5` để hạn chế conditioning quá mạnh khi đi vào G.

Đây là nguồn gốc của fix reference đặc biệt; không nên ghi nhận chúng là hoàn toàn mới trong commit 06/09.

### 3.3. Trạng thái hiện tại: reliability gate học từ visual evidence

Cơ chế ngày 06/09 dùng `image_height // 2` để ước lượng character width đã bị thay thế. Dù không kiểm tra trực tiếp `/`, `.` hay digit, heuristic đó vẫn gắn với hình học IAM và có thể đánh giá sai glyph hẹp của dataset khác.

[StyleEncoder.forward](./networks/module.py) hiện tính reliability riêng cho từng local token từ bốn nguồn visual evidence đã normalize:

```text
evidence = concat(global_token,
                  local_token,
                  abs(local_token - global_token),
                  masked_visual_feature_variance)
reliability = sigmoid(MLP(evidence))
local_stable = global_token + reliability × (local_token - global_token)
```

Gate không dùng image height, không ước lượng số character, không đọc character ID và không có danh sách punctuation. Vì vậy nó có thể học khi nào một local slot thực sự có evidence trên bất kỳ alphabet hoặc dataset nào, thay vì mặc định rằng reference hẹp luôn thiếu style.

### 3.4. Đồng bộ cả nhánh VAE

Reliability nói trên được áp dụng cho cả local mean và tensor local dùng để tính `logvar`, nên sampling VAE không thể bypass gate bằng cách đưa lại local variation chưa ổn định. Sau đó model vẫn dùng:

```text
logvar = clamp(logvar_head(style), −14, 4)
sample = mu + exp(0.5 × logvar) × noise
```

Như vậy cả mean conditioning lẫn đầu vào của variance head đều phản ánh mức hỗ trợ từ reference. Tuy nhiên, **norm cap 9.5 chỉ chặn mean; nó không bảo đảm norm của sample VAE luôn dưới 9.5**. Code áp cap cho mọi mean vector vượt ngưỡng, không có điều kiện chỉ áp cho punctuation. Đây là khác biệt cần giữ chính xác khi mô tả phương pháp.

### 3.5. Giới hạn thông tin vẫn còn

Một dấu chấm không thể xác định duy nhất writer viết `a`, `b`, `h` hoặc nối cả từ như thế nào. Cơ chế mới hướng tới đầu ra ổn định, hợp lý hơn khi thiếu evidence; nó không thể khôi phục chính xác mọi allograph từ một reference không chứa allograph đó. Global style vẫn được suy ra từ chính ảnh reference, không phải một reference dài bổ sung.

## 4. Sửa bug length scale x32

Trong `StyleBackbone` và `Recognizer`, convolution đầu của đường x32 dùng stride 1, còn x64 dùng stride 2. Trước commit `84dec47`, metadata vẫn coi cả hai đường có horizontal reduction 16.

Sau sửa:

| Thành phần | Trước: x32 | Sau: x32 | x64 |
|---|---:|---:|---:|
| `StyleBackbone.reduce_len_scale` | 16 | 8 | 16 |
| `Recognizer.len_scale` | 16 | 8 | 16 |

Khi độ dài hợp lệ bị chia cho 16 trong khi CNN chỉ giảm 8 lần, các consumer dùng metadata đó có thể chỉ sử dụng khoảng nửa feature sequence hợp lệ. Với style, đây là mất evidence; với OCR, CTC nhận độ dài đầu ra không đúng. Reference ngắn vốn ít dữ liệu lại càng nhạy với lỗi này.

Đây là **sửa tính đúng đắn của x32**, không phải tăng parameter hay thêm loss. Không được quy toàn bộ chênh lệch giữa DEV64 và MAIN32 cho Generator khi teacher hoặc length metadata từng sai. X64 vốn đã dùng scale 16 nên thay đổi này không trực tiếp tăng số timestep của x64.

Nguồn: [module.py](./networks/module.py), commit `84dec47`.

## 5. Rare-character learning: thay chuỗi ký hiệu nhân tạo bằng từ thật

### 5.1. Policy cũ và nguy cơ lệch phân phối

Từ ngày 30/08, `idx_to_words` có mặc định `rare_ratio=0.25`. Khi được chọn, code hoặc ghép một ký tự đặc biệt vào đầu/cuối từ, hoặc tạo chuỗi 1–5 ký tự ngẫu nhiên từ digits, punctuation và chữ hoa.

Policy này tăng cơ hội nhìn thấy ký tự hiếm nhưng cũng tạo nhiều chuỗi khác phân phối IAM. Generator cần học cả hình dạng chữ lẫn phân phối hình ảnh của từ. Quá nhiều tổ hợp phi tự nhiên có thể làm local statistics và global word statistics lệch khỏi real data, là một giả thuyết hợp lý cho KID kém dù ảnh trông sắc hơn. Chưa có ablation trong báo cáo này chứng minh mức độ tác động.

### 5.2. Policy mới trong `4753021`

- Giữ lexicon tiếng Anh cho độ đa dạng nội dung thông thường.
- Đọc transcription thật của training dataset để tạo `rare_lexicon`.
- Dùng `rare_word_ratio: 0.15` trong GAN32, GAN64 và local config.
- Tính tần suất ký tự trên corpus sau `casefold()`; trọng số ký tự tỉ lệ nghịch căn bậc hai tần suất.
- Điểm của một từ là trung bình trọng số các ký tự của từ đó. Khoảng 15% lựa chọn đi qua sampler có trọng số này.
- Capitalization và giới hạn độ dài vẫn được áp dụng sau bước chọn từ.

Điều này tăng xác suất chọn từ chứa ký tự ít gặp mà không tự tạo chuỗi punctuation ngẫu nhiên. Không nên mô tả là “mọi đầu ra text đều trùng nguyên văn corpus”: capitalization và cắt từ dài vẫn có thể thay đổi chuỗi sau sampling.

Do đếm sau `casefold`, policy không cân bằng riêng từng chữ hoa. `Q` và `q` cùng đóng góp vào một nhóm tần suất; chữ hoa còn phụ thuộc capitalization policy. Ngoài ra, một ký tự không xuất hiện trong pool không tự được bổ sung bằng cơ chế này.

### 5.3. Ảnh hưởng tới reference và evaluation

Rare sampler tăng cơ hội **nội dung đích** chứa ký tự hiếm. Nó không tự thay DataLoader để tăng số reference là `/`, `.`, `,`. Vì vậy không thể coi đây là fix duy nhất cho single-character style reference.

`sample_images` đặt `rare_ratio=0`, giúp ảnh sample không bị thay text bởi rare-word injection. Đường validation mặc định `use_rand_corpus: false` dùng transcription của dataset; nếu bật random corpus, code vẫn truyền rare ratio trong nhánh đó, cần ghi nhận khi so kết quả.

Nguồn: [_rare_lexicon_sampler và idx_to_words](./networks/utils.py), [khởi tạo rare_lexicon và các đường gọi](./networks/model.py), commit `4753021`.

## 6. StrokePatchD: giảm tác động của character label không chắc chắn

### 6.1. Vấn đề của vị trí ký tự ước lượng

Dataset cung cấp word boxes và transcription, không có bounding box chính xác cho từng glyph. Crop được đặt theo khoảng chia đều chiều rộng từ. Với handwriting, chữ `i` và `m` không rộng bằng nhau; ligature còn làm ranh giới khó xác định. Một patch mang label `a` chưa chắc chủ yếu chứa `a`.

Character-conditioned discriminator có thể gửi supervision nhiễu khi được yêu cầu chấm điểm theo một label thiếu chính xác. Đây là động cơ thêm confidence thay vì dùng mọi character projection với cùng cường độ.

### 6.2. Confidence mới

Sampler trả thêm confidence từ hai thành phần:

```text
geometry_confidence = mức giao giữa patch và span ký tự ước lượng
ink_fraction        = tỉ lệ pixel > −0.75 trong patch
ink_confidence      = clamp((ink_fraction − 0.005) / 0.04, 0, 1)
confidence          = geometry_confidence × ink_confidence
```

Theo convention nội bộ nền −1, ink fraction là một phép kiểm tra lượng nét hiện diện. GAN config bật `patch_char_conditioning: true`, với `patch_char_min_confidence: 0.25`. Confidence dưới ngưỡng được đưa về 0.

Trong P, điểm số có dạng:

```text
score = unconditional_score + confidence × character_projection
```

Patch ít ink hoặc thiếu hỗ trợ hình học vẫn đóng góp vào adversarial loss qua unconditional score, nhưng character projection yếu đi hoặc tắt. Policy được dùng cho fake ngẫu nhiên, style-transfer fake, reconstruction fake, real và augmented-real.

### 6.3. Những giới hạn cần nêu đúng

Confidence là heuristic hình học và lượng ink, không phải xác suất OCR rằng crop chứa đúng ký tự. Nó không giải quyết hoàn toàn sai lệch alignment do độ rộng chữ không đều. Các dấu nhỏ thật như `.` có thể bị giảm character conditioning vì lượng ink thấp; unconditional path vẫn còn, nhưng đây là trade-off cần kiểm tra bằng tập punctuation riêng.

Confidence hiện được tính trước optional patch masking. Vì vậy với `masking_mode` bật, confidence phản ánh patch trước mask, không hoàn toàn phản ánh lượng ink còn lại sau mask.

Patch vẫn thích nghi theo độ rộng: `ceil(valid_width / patch_size)` chặn trong `[4, 8]`; chia vùng chọn theo các khoảng chỉ số ký tự và luân phiên vùng dọc trên/dưới. GAN64 dùng patch 32×32, GAN32 dùng 16×16. Crop policy được ghép tương ứng giữa real/fake, nhưng không có nghĩa tất cả ảnh dùng cùng tọa độ ngẫu nhiên.

Nguồn: [sample_character_patches](./networks/utils.py), [PatchDiscriminator.forward](./networks/BigGAN_networks.py), [prepare_stroke_patches](./networks/model.py), commit `4753021`.

## 7. Các thay đổi hướng đến KID

Có hai nhóm cần tách khi báo cáo: thay đổi quá trình học và sửa cách đo.

### 7.1. Quá trình học

Rare-word sampling theo corpus giảm việc ép G học nhiều chuỗi nhân tạo. Confidence trong StrokePatchD giảm áp lực character conditioning khi crop không rõ. Short-reference stabilization hạn chế biến thiên local thiếu evidence. Đây là những cơ chế có khả năng cải thiện phân phối nét và từ, từ đó có thể giúp KID; không cơ chế nào bảo đảm KID giảm trong mọi run.

### 7.2. Cache ảnh evaluation chuyển từ int8 sang float32

Trước đây ảnh fake cache trên CPU bị lượng tử hóa:

```text
stored = round(clamp(image, −1, 1) × 127).to(int8)
read   = stored.float() / 127
```

Commit `4753021` thêm `valid.metric_cache_dtype`, mặc định/config hiện tại là `float32`; `float16` và `int8` vẫn là lựa chọn khác. Float32 giữ precision của output FP32 trong khoảng clamp, tránh bước làm tròn int8 trước khi trích feature.

Trong khoảng chuẩn hóa, bước int8 là `1/127`; sai số làm tròn tối đa khoảng `1/254` nếu không xét clipping. Nó có thể ảnh hưởng các thay đổi cường độ nét nhỏ, nhưng code không chứng minh đây là nguyên nhân chính của KID cao. Việc giữ precision làm phép đo sát output hơn; **không phải huấn luyện model tốt hơn, và điểm có thể tăng hoặc giảm**.

Chi phí là phần tensor ảnh trong host cache dùng khoảng 4 lần dung lượng int8, chưa tính overhead và các tensor khác. `float32` ở đây là dtype cache evaluation, không phải thêm FP16 training.

### 7.3. Cache hợp lệ và thiết bị tính KID

Code thêm key cho validation DataLoader theo dataset, split, batch size và image height; real statistics cache cũng có key theo dataset hiện tại, feature dimensions, crop mode và nhu cầu IS. Một số cache HWD/CMMD được reset khi tạo lại loader. Cách này giảm nguy cơ tái sử dụng real features của cấu hình cũ trong cùng process. Nó không phải fingerprint nội dung HDF5: thay file tại chỗ mà không đổi key chưa được bảo đảm phát hiện.

`polynomial_mmd_averages` và `polynomial_mmd` nhận `device` từ evaluator, tránh tự dùng GPU mặc định khi caller chọn thiết bị khác. Polynomial kernel KID không được thay bằng một công thức dễ đạt điểm thấp hơn; config vẫn degree 3, 50 subsets và subset size 1000.

### 7.4. Lưu KID để theo dõi

Checkpoint thêm `kid` và `last_eval_kid`, có đường restore để tiếp tục giữ metric gần nhất. Việc chọn **best checkpoint vẫn theo FID**, không tạo cơ chế chọn best-KID riêng. Eval history/CSV là cơ chế đã có từ trước; commit này bổ sung tracking KID chứ không nên ghi là mới tạo toàn bộ CSV ledger.

Nguồn: [AdversarialModel.validate và BaseModel.save/load](./networks/model.py), [metric/val_metrics.py](./metric/val_metrics.py), commit `4753021`.

## 8. Sửa cấu hình StyleEncoder từng bị bỏ qua

Trước `4753021`, nhiều khóa YAML lọt vào `**kwargs` và không được áp dụng. Đây là vấn đề khiến file config mô tả một experiment nhưng code chạy khác.

| Khóa | Cách xử lý mới |
|---|---|
| `num_style_tokens`, `num_local_queries` | Ràng buộc tổng token = local + 1; suy ra số token nếu chỉ có local |
| `query_dim` | Kiểm tra bằng `in_dim`; không giả vờ hỗ trợ projection dimension riêng |
| `heads` | Dùng trực tiếp cho MultiheadAttention, kiểm tra tính chia hết |
| `cross_attn_dropout` | Truyền vào attention |
| `local_attention_gate_init` | Hỗ trợ alias; GAN32/64 đổi sang tên chính `local_attention_residual_init` |
| `feature_scales` | Chọn đúng feature map tương ứng ID 1/2/4 |
| Khóa không biết | Báo lỗi thay vì im lặng bỏ qua |

GAN32/64 ghi rõ 8 total tokens, 7 local queries, 4 heads, attention dropout 0, local attention residual init 0.5. Trước sửa, `local_attention_gate_init: 0.5` có thể bị bỏ qua và constructor dùng mặc định 0.25. Sửa này thay đổi initialization khi retrain; khi load weights đã có gate logits, giá trị trong checkpoint mới quyết định trạng thái gate.

GAN32 đổi `feature_scales: [1, 2]` thành `[1, 2, 4]` để giữ cấu trúc ba projection tương thích weights trước đó. Vì config cũ bị bỏ qua nên không nên diễn giải đơn giản là vừa tăng từ hai lên ba scales trong mạng thực chạy. Code cũng đồng bộ số style tokens hiệu lực vào các đường random style/interpolation để tránh dùng fallback 32 tokens không khớp encoder 8 tokens.

Nguồn: [StyleEncoder.__init__](./networks/module.py), [GlobalLocalAdversarialModel và interpolation](./networks/model.py), [GAN64 config](./configs/gan_iam_64.yml), commit `4753021`.

## 9. Giảm tính toán dư thừa và áp lực bộ nhớ

### 9.1. Tái sử dụng backbone feature

Trong một iteration, backbone B đã frozen và ở eval mode. Feature của real style reference được tính một lần, sau đó dùng ở pha D và pha G. Encoder E vẫn chạy lại ở pha G với gradient; không cache output E rồi vô tình cắt đường học.

Với style-transfer image, encoder trả thêm backbone feature đã tính. WriterIdentifier có `forward_from_feat`, giúp W dùng lại feature đó thay vì gọi B lần nữa. Đường gradient từ writer loss về ảnh generated vẫn được giữ.

Contextual loss cũng dùng lại intermediate features từ E. Cần lưu ý các feature này có width masking; đây có thể thay đổi xử lý padding so với đường W trước đó, nên không nên tuyên bố toàn bộ refactor là bit-for-bit equivalent.

### 9.2. Pool trước projection

Các feature maps được average-pool chiều cao xuống 4 hàng trước khi chạy pointwise 1×1 projection. Với affine projection và average pooling, hai phép toán giao hoán về toán học, nên có thể giảm số vị trí cần chiếu mà vẫn giữ cùng biểu thức, ngoại trừ sai khác floating-point. Mức tiết kiệm phụ thuộc kích thước từng feature map.

### 9.3. Vector hóa patch extraction

Crop sampler chuyển từ vòng lặp Python và scalar lấy về CPU sang các phép tính tensor, rồi gather đúng pixel của patch cần lấy. Cách này giảm overhead gọi kernel và tránh dùng unfold backward trên mọi sliding window. Tuy nhiên vẫn có thao tác như `nonzero` với số crop động; không có cơ sở nói code đã loại bỏ mọi đồng bộ GPU–CPU hay mọi biến động VRAM.

### 9.4. Logging và gradient buffers

`AverageMeterManager.update_many` gom nhiều scalar đã detach vào một lần chuyển CPU, thay vì gọi `.item()` cho từng loss. Tra lexicon cũng chuyển cả batch chỉ số sang CPU một lần. Sau optimizer step, D/P và G giải phóng gradient buffers bằng `zero_grad(set_to_none=True)` khi không còn cần.

Rare sampler cache cumulative distribution rồi dùng tìm kiếm trên CDF, tránh tính lại xác suất cho toàn corpus ở mỗi từ.

Đây là những lý do kỹ thuật để kỳ vọng training hiệu quả hơn. Chưa có benchmark trước/sau trên cùng T4, batch size và word-length distribution trong báo cáo này, nên không đưa ra hệ số tăng tốc.

Nguồn: [networks/model.py](./networks/model.py), [networks/module.py](./networks/module.py), [networks/utils.py](./networks/utils.py), [lib/utils.py](./lib/utils.py), commit `4753021`.

## 10. Pretraining, vận hành và các sửa lỗi khác

### 10.1. Cập nhật teacher checkpoints

Các commit `597d65b`, `91b107d` và `4753021` cập nhật OCR/W checkpoints x32 hoặc x64. Đây là thay đổi thực nghiệm đáng ghi vì supervision của GAN phụ thuộc teacher. Chỉ diff file nhị phân không cho biết teacher tốt hơn bao nhiêu; báo cáo này không dùng số metric cũ để gán thành tích cho mỗi lần thay weights.

Việc xóa `style_prior_mean.npy` trong `597d65b` cũng không đủ để khẳng định một learned prior mới đã được thêm vào encoder. Cần phân biệt file tài nguyên với cơ chế đang chạy.

### 10.2. RESULTS table và log lúc khởi chạy

Commit `1f48857` thay việc in toàn bộ config bằng bản tóm tắt: model, dataset, resolution, device, batch/epochs/workers, learning rates, critic steps, metrics và parameter counts theo network. Config đầy đủ vẫn được ghi riêng.

Khi kết thúc, GAN/OCR/W có RESULTS table với trạng thái, thời gian, epoch đã hoàn thành, cấu hình cơ bản, checkpoint và metric. `train.py` xử lý completed/failed/interrupted và xuất status record bằng ghi file tạm rồi replace. W&B được đóng sau bước ghi kết quả. Đây là cơ sở để notebook monitor biết job đã kết thúc, nhưng chỉ sửa model/status chưa chứng minh mọi phiên bản notebook đều tiêu thụ status đúng.

### 10.3. Tensor subclass và resume metadata

`Distribution.to()` kiểm tra sampler metadata. Tensor phát sinh qua phép toán nhưng vẫn mang subclass, không còn `dist_type`, được chuyển về tensor thường. Điều này tránh nhầm phương thức `Tensor.var`/`mean` với các thuộc tính số của distribution, loại lỗi từng có thể làm attention path crash khi chuyển dtype/device.

Resume xử lý `best_fid` cẩn thận hơn trước khi format và khôi phục KID gần nhất. Đây là cải thiện độ bền vận hành, không phải thay đổi năng lực Generator.

## 11. Những gì không phải thay đổi mới trong 01–10/09

- [4a9f0be, 31/08](https://github.com/rquq/HGGAN_test/commit/4a9f0be9feccbc1a37d978a5caef99f7cae6660d) đã đổi `num_critic_train` từ 1 sang 2 và tắt IS trong GAN32/64. Đây là bối cảnh ngay trước kỳ, không phải thay đổi ngày 06/09. Với code hiện tại, D/P update mỗi batch và G update theo chu kỳ critic, nên số optimizer steps của G trên mỗi epoch cần được xét khi so run.
- Global D chỉ nhận whole-word inputs, local crops đi vào P là thiết kế đã tồn tại; commit mới điều chỉnh confidence/crop implementation, không mới tạo sự phân chia đó.
- Fusion/allograph stack không có diff trong năm commit của kỳ. Không nên ghi nhận một kiến trúc fusion mới hay thay toàn bộ G trong giai đoạn này.
- Các loss CTC random/style có schedule riêng, reconstruction decay, EMA, content-adversarial probe và orthogonal query initialization không phải tất cả vừa được thêm trong kỳ này. Chúng có trong code hiện tại nhưng cần lịch sử xa hơn nếu muốn báo cáo nguồn gốc.

## 12. Cách trình bày thành quả và xác nhận tiếp theo

Phát biểu phù hợp với bằng chứng hiện có:

> Trong giai đoạn 01–10/09/2026, DEV được cải tiến nhằm ổn định style conditioning khi reference có ít thông tin, đặc biệt là dấu câu và ký tự đơn. Encoder kết hợp độ rộng ảnh và feature sequence để kiểm soát local style, đồng thời áp dụng cùng cơ chế vào thống kê VAE. Training text chuyển sang ưu tiên từ thật chứa ký tự hiếm; StrokePatchD giảm character conditioning trên các crop thiếu evidence. Quá trình đánh giá giữ cache float32 và quản lý cache/device rõ hơn để đo KID nhất quán. Các tối ưu chia sẻ backbone features, vector hóa crop và gom logging giảm công việc dư thừa trong training.

Để có thể bổ sung phát biểu “cải thiện chất lượng đã được chứng minh”, cần so cùng resolution, data, teacher, ngân sách G steps và evaluator. Tập reference nên tách `/`, `.`, `,`, `-`, một chữ cái, một chữ số và từ bình thường; mỗi reference sinh cùng danh sách target words và cùng seed. Ngoài FID/KID toàn tập, nên kiểm tra readability và lỗi kéo dài/nhân nét trên riêng nhóm reference ngắn.

Nên chạy lại cả checkpoint cũ và mới bằng cùng cache float32. Nếu chỉ so một điểm KID cũ dùng int8 với điểm mới dùng float32, chênh lệch vừa có thể do model vừa có thể do đường đo. Muốn biết thành phần nào đóng góp, các ablation hữu ích là: support mới, rare sampler mới, và patch confidence, mỗi lần chỉ thay một yếu tố. Những thử nghiệm này là đề xuất xác nhận, chưa được chạy trong phiên lập báo cáo.

## 13. Thay đổi nhỏ trong phiên lập báo cáo

Tiêu đề startup trong `BaseModel.info()` của DEV được đổi từ `HGGAN RUN` sang tên thư mục checkout chứa model, tức `dev` ở nhánh hiện tại. Cách lấy tên dựa trên vị trí file code nên vẫn đúng khi gọi từ working directory khác và khi notebook checkout ở trạng thái detached HEAD; nó không cần chạy Git mỗi lần startup. Nếu tự đổi tên thư mục checkout thì tiêu đề theo tên thư mục đó.

Thay đổi này áp dụng cho các model trong DEV dùng chung `BaseModel.info()`. Đây là sửa giao diện log trong phiên hiện tại, chưa thuộc năm commit đã phân tích ở trên.

## 14. Texture generator được tạo khi nào, và DEV có dùng không?

### 14.1. Nguồn gốc xác nhận từ Git

| Nhánh | Commit đầu tiên thêm file | Thời gian UTC+7 | Điều gì thực sự xảy ra |
|---|---|---|---|
| MAIN | [cdf36e2](https://github.com/rquq/HGGAN_test/commit/cdf36e299aef5266f30c685bb106804ac8f7d77b) | 18/08/2026 14:09:25 | Thêm texture module; cùng commit bỏ `rapidnet.py` và thay đổi G |
| DEV | [009014e](https://github.com/rquq/HGGAN_test/commit/009014e4e67c147710c65d12db6ad0a137b8efc6) | 30/08/2026 23:02:24 | Thêm module vào DEV và nối trực tiếp vào output path của G |
| DEV | [87fe941](https://github.com/rquq/HGGAN_test/commit/87fe941d6d5681281111e05cad547dcc3aad9638) | 30/08/2026 23:44:34 | Chỉ sửa một comment trong texture file; không thay computation của module |

Docstring đầu file vẫn viết “MAIN's generator”. Đó là mô tả còn sót lại từ nguồn gốc MAIN, **không có nghĩa DEV bỏ qua module**.

Call path trong DEV hiện tại:

```text
train.py → GlobalLocalAdversarialModel
         → networks.BigGAN_networks.Generator
         → self.texture_refinement = StyleFrequencyRefinement(...)
         → Generator.forward: texture_detail = self.texture_refinement(h, z)
         → output = tanh(base_logits + texture_detail)
```

Các điểm nối được xác nhận ở [networks/model.py](./networks/model.py), [BigGAN_networks.py](./networks/BigGAN_networks.py) và [texture_generator.py](./networks/texture_generator.py). GAN32 và GAN64 đều khởi tạo cùng class Generator này; không thấy config toggle tắt texture branch. Như vậy file là dependency đang hoạt động, không phải backup hay file test thừa.

**Cần điều chỉnh cách mô tả DEV:** từ 30/08, G của DEV là GBlock backbone **cộng** StyleFrequencyRefinement ở output. Nếu gọi DEV hiện tại là “classic GBlock thuần, không có texture branch” thì không còn đúng. Nói rằng fusion không đổi trong 01–10/09 vẫn đúng, nhưng không được suy rộng rằng G chưa từng thay trong toàn khoảng 30/08–10/09.

### 14.2. Đây không phải một Generator độc lập

Tên file dễ gây hiểu nhầm. Nó không nhận text rồi tự sinh cả ảnh, không thay cả GBlock stack, không có optimizer riêng hay loss riêng. Nó là một **nhánh refinement trainable nằm cuối Generator**, nhận:

- `h`: feature map sau GBlock cuối, đã ở chiều cao ảnh output.
- `z`: bộ style tokens từ encoder hoặc random style sampler.

Đầu ra là một residual cùng kích thước với image logits. Residual được cộng vào logits trước `tanh`, vì vậy nó tham gia vào ảnh dùng cho discriminator, OCR, style/writer và các loss khác qua cùng G graph.

## 15. Cơ chế của StyleFrequencyRefinement

### 15.1. Tóm tắt style distribution

Với config hiện tại, mỗi sample có 8 tokens × 32 chiều. Descriptor gồm:

```text
descriptor = concat(global_token, mean(local_tokens), std(local_tokens))
           = vector 96 chiều
```

Global token mô tả writer-level style. Mean/std của bảy local tokens cung cấp thống kê tổng quát về phần local style. “Style distribution” ở đây chỉ là hai moments trên token slots, không phải một mô hình mật độ đầy đủ, cũng không có bảo đảm các slots là những mẫu độc lập từ một distribution.

Descriptor không giữ thứ tự local slots và không dùng character ID của từng vị trí. Do đó nhánh này không trùng hoàn toàn với allograph refinement ở fusion: fusion điều chỉnh content theo ký tự; texture branch điều chỉnh chi tiết cuối ảnh theo style statistics. Dù vậy, cả hai cùng tác động nét chữ nên vẫn có thể dư thừa một phần; chỉ ablation mới cho biết nhánh thêm có đem lại lợi ích thực tế.

### 15.2. Ba nhánh convolution theo hướng

Feature map `h` đi qua ba depthwise convolutions:

| Nhánh | Kernel | Vai trò dự kiến |
|---|---|---|
| Horizontal | 1×7 | Nét và cạnh theo phương ngang |
| Vertical | 7×1 | Nét và cạnh theo phương dọc |
| Local | 3×3 | Chi tiết cục bộ gọn |

`branch_affine(descriptor)` tạo ba weights cho mỗi channel, chuẩn hóa bằng softmax. Vì thế mỗi sample có thể chọn tỷ lệ khác nhau giữa ba hướng. Đây không phải attention giữa toàn bộ pixel; weights là theo sample/channel và dùng chung trên không gian.

Sau tổng có trọng số, feature đi qua SiLU, channel-mixing convolution 1×1, rồi Global Response Normalization (GRN). GRN dùng L2 response theo không gian, chuẩn hóa tương đối giữa channels, với gamma/beta học được.

### 15.3. Style modulation có giới hạn

Một linear head khác tạo scale và shift:

```text
scale_raw = 1 + 0.15 × tanh(scale_head(descriptor))
scale     = scale_raw / RMS(scale_raw)
shift     = 0.15 × tanh(shift_head(descriptor))
```

Feature được biến đổi bằng scale/shift rồi chiếu xuống một kênh detail logits. Giới hạn ±0.15 áp vào biểu thức scale trước RMS normalization và vào shift; không nên nói scale cuối luôn nằm chính xác trong [0.85, 1.15].

Lúc reset, hai affine heads và GRN gamma/beta được zero-init: weights ba nhánh bằng nhau, modulation gần identity và GRN là identity. Spatial convolutions vẫn có weights, residual gain ban đầu khác 0, nên **toàn texture branch không phải identity hoàn toàn tại initialization**.

### 15.4. Tách high-frequency rồi cộng vào output

```text
raw_detail  = detail_head(refined_features)
low_detail  = average_pool_3×3(reflect_pad(raw_detail))
high_detail = raw_detail − low_detail
gain        = 0.15 × sigmoid(detail_gain_logit)
image       = tanh(base_logits + gain × high_detail)
```

Gain khởi tạo là `0.03`, giới hạn trên `0.15`. High-pass làm nhánh thiên về chi tiết cạnh/nét thay vì thêm trực tiếp một vùng sáng tối chậm biến thiên.

Tuy nhiên, **gain nhỏ không phải bound tuyệt đối cho residual**, vì `high_detail` không được hard-clamp theo amplitude. “Không thay đổi geometry” trong docstring là mục tiêu thiết kế, không phải bảo đảm toán học: cộng residual trước `tanh` vẫn có thể làm nét mảnh rõ hơn, đứt hơn, xuất hiện viền hoặc thay đổi vùng gần ngưỡng. Phép high-pass với biên reflect và phi tuyến sau đó cũng không bảo đảm toàn bộ low-frequency của ảnh cuối được giữ nguyên.

File ghi chú ý tưởng từ ConvNeXt-V2/GRN và gating nhiều nhánh gợi từ MogaNet/InceptionNeXt. Đây là module kết hợp trong repository; không nên mô tả là bê nguyên một architecture từ một paper. Báo cáo này xác nhận implementation và Git history, không đánh giá lại mức độ trung thành với từng paper.

## 16. Nó có cần thiết không, và chi phí có thật sự nhẹ không?

### 16.1. Cần để chạy code hiện tại: có

Generator import, instantiate và gọi module vô điều kiện. Xóa riêng file sẽ gây lỗi import; bỏ call còn đòi hỏi thay graph và xử lý state của checkpoint. Với model đã train có nhánh này, output của nó đã tham gia tối ưu chung; không nên tháo bỏ rồi mặc định giữ nguyên chất lượng.

### 16.2. Bắt buộc về mặt phương pháp GAN: không

GBlock base vẫn có output head hoàn chỉnh. Về kiến trúc, hoàn toàn có thể xây một variant chỉ dùng `tanh(base_logits)` để đối chứng. Nhánh texture là một lựa chọn thực nghiệm tăng khả năng biểu diễn, không phải điều kiện để mô hình học sinh handwriting.

Nó cũng không phải cơ chế chính giải quyết reference `/` hay `.`. Nhánh này nhìn style tokens của chính reference thiếu thông tin; không tạo được bằng chứng writer mới. Khi local mean/std bất ổn, nó còn có thể truyền bất ổn đó vào chi tiết output. Vì vậy support factor ở encoder vẫn là phần cần thiết để giải quyết vấn đề upstream.

### 16.3. Nhẹ về parameters, chưa chắc nhẹ về activation

Với `channels=64`, `style_dim=32`, `output_channels=1`, đã đếm trực tiếp module được khởi tạo: **37,058 trainable parameters**. Spectral-normalization buffers không tính là parameters; convolution parameter shapes không đổi khi dùng SNConv2d của G.

| Thành phần | Parameters |
|---|---:|
| Depthwise ngang + dọc + local | 1,664 |
| Channel mixing 1×1 | 4,160 |
| GRN | 128 |
| Style affine | 12,416 |
| Branch affine | 18,624 |
| Detail head | 65 |
| Gain logit | 1 |
| Tổng | **37,058** |

Nhánh chạy ở full spatial resolution, nên overhead không thể suy ra chỉ từ 37 nghìn parameters. Với một ảnh 64×320, riêng năm convolutions tương ứng khoảng **115.34 triệu MACs**; chưa tính gating, normalization, activation, spectral normalization và backward. Đây là ước tính toán học theo kernel/channel, không phải số đo thời gian trên GPU.

Ở batch 8, riêng ba output tensors của ba depthwise branches tại 64×320, 64 channels, FP32 đã có tổng dung lượng khoảng **120 MiB**. Đây không phải peak memory của cả refiner: autograd, tensors trung gian, lifetime và allocator còn ảnh hưởng. Gọi G trên nhiều nhóm ảnh ghép batch cũng làm chi phí tăng theo effective batch.

Vì vậy đánh giá đúng là: **ít parameters nhưng có chi phí compute/activation ở cuối G**. Khi width và height cùng tăng gấp đôi, số pixel tăng bốn lần và chi phí spatial của nhánh tăng tương ứng ở cùng batch size.

### 16.4. Khuyến nghị dựa trên bằng chứng hiện có

Giữ nó trong checkpoint/model hiện tại trong lúc đánh giá. Chưa có bằng chứng đủ để nói module bắt buộc giúp KID, cũng chưa có bằng chứng đủ để kết luận nó làm KID xấu đi. Không nên xóa chỉ vì tên file khác với tên G.

Một phép thử trên cùng checkpoint có thể đo output có/không có residual, giữ nguyên seed/reference/target text để xem model hiện tại phụ thuộc nhánh này bao nhiêu. Kết quả đó là **inference ablation**, không tương đương so sánh hai model được train riêng từ đầu. Nếu cần quyết định loại khỏi kiến trúc để ưu tiên tốc độ, nên tiếp tục bằng training ablation cùng ngân sách G updates và evaluator float32.

Cũng nên đo `RMS(gain × high_detail)` so với `RMS(base_logits)`, vì chỉ nhìn gain không biết đóng góp thực. Không thêm những diagnostic này vào W&B trong phiên hiện tại; đây là đề xuất cho phép đo cục bộ nếu thực hiện ablation.

### 16.5. Kiểm tra thực hiện trong phiên bổ sung

- Đọc đường import/khởi tạo/forward của G trong DEV và MAIN.
- Xác nhận commit đầu tiên thêm file ở cả hai nhánh.
- Đếm parameter của standalone refiner bằng PyTorch.
- Chạy CPU FP32 forward/backward ở chiều cao 32 và 64, batch 2, width 96 với Conv2d thường cùng kernel/group cấu hình. Output shape và giá trị hữu hạn hợp lệ; gradient về feature map có giá trị khác 0.
- Tại identity initialization, gradient từ nhánh về style tokens bằng 0 do affine weights khởi tạo 0; affine heads vẫn là tham số trainable. Không nên tuyên bố texture branch truyền style gradient khác 0 ngay từ bước đầu.

Các kiểm tra này không phải full training test, không benchmark T4 và không đo FID/KID mới. Không chỉnh computation của texture module trong phiên này.

## 17. Timeline đầy đủ từ commit thêm texture vào DEV

Đây là 11 commit trước cửa sổ 01–10/09, nối tiếp bằng 5 commit đã phân tích ở phần 2. Các mục ghi rõ thay đổi cấu trúc, thay đổi vận hành và dọn tài liệu để tránh coi mọi commit là cải tiến chất lượng.

| Thời gian UTC+7 | Commit | Nội dung và ý nghĩa |
|---|---|---|
| 30/08 23:02 | [009014e](https://github.com/rquq/HGGAN_test/commit/009014e4e67c147710c65d12db6ad0a137b8efc6) | Thêm texture refiner và nối vào G; thêm support factor/norm cap ở E; thêm special-character injection cũ; tách GAN32/GAN64 config. Trong local config: critic 2→1, patch weight 0.4→0.45, min LR ratio 0.25→0.15, prefetch 2→3. Các thay đổi này đến cùng nhau nên metric sau commit không cô lập riêng tác dụng texture. |
| 30/08 23:29 | [6865389](https://github.com/rquq/HGGAN_test/commit/68653898e19755c123a6e943af42a7653c52485d) | Thêm G_arch32 gồm ba stages; G64 dùng bốn stages. Thêm config OCR/W32; D/P nhận thêm kwargs; load W pretrained chuyển strict=False trong GAN path. Đây vừa là hỗ trợ resolution vừa có trade-off kiểm tra checkpoint. |
| 30/08 23:34 | [63ea863](https://github.com/rquq/HGGAN_test/commit/63ea863a68e17898628941718eecabf064a3ba65) | GAN32 trỏ đúng `wid_iam_train32.pth` và `ocr_iam_train32.pth`, thay tên pretrained chung. |
| 30/08 23:37 | [81e1f96](https://github.com/rquq/HGGAN_test/commit/81e1f9633abdae53274323d46a07f20869d3c578) | Dọn config dump, báo cáo/research artifacts và PDF. Không phải thay đổi thuật toán GAN. |
| 30/08 23:44 | [87fe941](https://github.com/rquq/HGGAN_test/commit/87fe941d6d5681281111e05cad547dcc3aad9638) | Chuẩn hóa tên deploy/checkpoint và comment; deploy ưu tiên EMA nếu có, thêm tùy chọn raw. Trong texture file chỉ đổi comment. EMA export có thể ảnh hưởng chất lượng inference được quan sát dù training graph không đổi. |
| 30/08 23:59 | [01761f8](https://github.com/rquq/HGGAN_test/commit/01761f8d577b1ad619eba4f1a7f9f1eb07de4b47) | Truyền img_height vào B/R và điều chỉnh convolution đầu theo x32/x64 trong GAN/pretraining/eval. Nhưng length-scale metadata chưa đổi tương ứng; lỗi còn này được sửa ở `84dec47` ngày 06/09. |
| 31/08 00:05 | [08b947d](https://github.com/rquq/HGGAN_test/commit/08b947d4f163e68a2ee6a838d8053cfc0034aa7e) | Backbone squeeze khi height=1, nếu còn nhiều hàng thì mean theo height để trả sequence đúng dạng. Bỏ ép `PYTORCH_CUDA_ALLOC_CONF=backend:native` trong train.py. |
| 31/08 00:27 | [4173e5a](https://github.com/rquq/HGGAN_test/commit/4173e5acd102d125d46c550d9614daddd1bfe16f) | Tăng batch W32/W64 từ 128 lên 256; điều chỉnh ngân sách bộ nhớ/tốc độ pretraining. |
| 31/08 11:20 | [2b86d43](https://github.com/rquq/HGGAN_test/commit/2b86d43fecbbf7e023e473ec3f1ea9df2b1f7402) | Đưa batch OCR/W32/64 về 128. Vì có commit điều chỉnh sau, không nên báo cáo batch 256 là cấu hình cuối giai đoạn này. |
| 31/08 15:29 | [69b1e35](https://github.com/rquq/HGGAN_test/commit/69b1e354555519aeb4ed4da92013a69b1242a325) | Đồng bộ dataset paths với img_height lúc chạy; resize style theo h thực tế và h/2 per-character width; transforms/padding lấy scale hiện hành thay vì default đóng băng lúc import. Giảm nguy cơ GAN32 dùng assumptions x64. |
| 31/08 15:41 | [4a9f0be](https://github.com/rquq/HGGAN_test/commit/4a9f0be9feccbc1a37d978a5caef99f7cae6660d) | Đổi critic 1→2 ở GAN configs; tắt IS ở GAN32/64. Không còn giữ critic=1 của ngày 30/08. |

Chuỗi phát triển có ba bước lớn: **30/08 đưa texture refinement vào DEV và mở hỗ trợ x32; 31/08 chỉnh data/config/runtime cho đa resolution; 06/09 sửa phần còn sai và tinh chỉnh short-reference, sampling, patch conditioning, metrics cùng tốc độ.** Vì vậy khi giải thích một lần chất lượng tăng/giảm, cần xét cả teacher, data geometry và critic update budget thay vì gán hết cho texture branch.
