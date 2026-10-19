# ALPHA so với HiGAN+: thiết kế, contribution và bằng chứng

**Báo cáo kỹ thuật — 15/10/2026**  
Đối tượng: nhánh `alpha` trong HGGAN_test. Ngôn ngữ: tiếng Việt; giữ thuật ngữ tiếng Anh khi giúp đọc chính xác hơn.

## Tóm tắt

ALPHA phát triển từ HiGAN+ theo hướng **tách ba trách nhiệm: biểu diễn style, dựng nội dung và giám sát nét chữ**. Thay vì nén toàn bộ reference vào một vector rồi lặp lại cho mọi ký tự, phiên bản hiện tại giữ một global code và bổ sung bảy local evidence tokens. Generator dựng nền cấu trúc từ target text trước; global style điều kiện hóa các residual blocks, còn local evidence được truy xuất tại các vị trí trên feature map. Một StrokePatchD mới đánh giá nét và liên kết cục bộ bằng các crop có kích thước suy ra từ chiều cao ảnh, song song với global discriminator nhìn toàn từ.

Ba thay đổi tạo thành một thiết kế thống nhất: **E giữ thông tin nào, G sử dụng thông tin đó ở đâu, và D/P kiểm tra sản phẩm ở cấp độ nào**. Đây là contribution ở cấp hệ thống và cách tổ chức representation; không nên tuyên bố attention, depthwise convolution, conditional normalization hay patch discrimination là phát minh mới.

Checkpoint ALPHA ngày 14/10 đạt FID **2.446330**, KID **0.063558** và HWD **0.093104** tại epoch 74, trên run native 32px. Tuy nhiên, checkpoint này có encoder cũ, trước đợt đơn giản hóa ngày 14/10. Báo cáo vì vậy phân biệt rõ **kết quả lịch sử đã đo** với **kiến trúc hiện tại chưa có kết quả retrain hoàn chỉnh**. Các số liệu cũ không chứng minh riêng tác động của từng module mới.

## 1. Phạm vi, phiên bản và cách đọc

Bản code được đối chiếu trực tiếp:

| Đối tượng | Nguồn và phiên bản | Vai trò trong báo cáo |
|---|---|---|
| ALPHA hiện tại | `alpha`, commit `7eed74c6b2d5286dc7b3e446ec1db3b165322c39` | Nguồn chính để mô tả architecture và loss |
| HiGAN+ gốc | Checkout `HiGANplus`, commit `5d37de733f79e2a72a79c0f66e1865329ff56b4e` | Đối chứng kiến trúc; không dùng classic_optimized thay cho bản gốc |
| Checkpoint 14/10 | `chkpoints/14_10_2026/*.pth` và lịch sử eval nhúng | Kết quả của phiên bản trước thay đổi E mới nhất |
| Checkpoint 07/10 | `chkpoints/7_10_2026/*.pth` | Bối cảnh thí nghiệm; không nối thành cùng một trajectory |

Các tensor được ký hiệu: `x_ref` là reference giữ hình học gốc; `x` là ảnh thật đã được chuẩn hóa cho đường training; `y` là transcript của ảnh thật; `y′` là text khác được lấy từ corpus; `w` là writer identity; `g` là global style; `L` là tập local tokens. `B` là backbone, `E` là style encoder học trong GAN, `G` là generator, `D` là global discriminator, `P` là StrokePatchD, `R` là OCR, `W` là writer head.

**Ba mức khẳng định được dùng xuyên suốt:**

- **Xác nhận từ code:** module, tensor, công thức hoặc đường gradient hiện hữu trong source.
- **Động cơ thiết kế:** lý do hợp lý cho một thay đổi; chưa tự động đồng nghĩa có cải thiện metric.
- **Bằng chứng thực nghiệm:** số đo từ checkpoint/log có nguồn; chỉ suy luận trong giới hạn protocol và phiên bản tương ứng.

Toàn bộ công việc ở đây là đọc và viết báo cáo. Không train lại, không chạy lại full metrics, không sửa model/config/dataset, và không thay đổi Git branch.

## 2. HiGAN+ đã làm được gì trước khi có các thay đổi?

HiGAN+ đã hỗ trợ generation theo text với style ngẫu nhiên hoặc style lấy từ một reference. Trong code gốc, backbone B tạo feature sequence; E dùng valid-width mean pooling, MLP và hai đầu `mu/logvar` để tạo một vector style 32 chiều. G ghép vector này với embedding của từng ký tự để dựng seed, rồi render ảnh qua bốn GBlocks với conditional batch normalization. Một phép chiếu style tạo conditioning riêng cho các blocks. [P1]

HiGAN+ **đã có cả D nhìn toàn ảnh và P nhìn patch**. Nó cũng đã dùng OCR/CTC, writer classification, latent reconstruction, pixel reconstruction, contextual matching và KL tùy chế độ. Vì vậy, những câu như “đề xuất dual discriminator”, “lần đầu dùng writer loss” hoặc “thêm contextual loss để so sánh style khác text” sẽ không phản ánh đúng contribution của dự án.

Điểm có thể cải tiến nằm ở các lựa chọn cụ thể của baseline: một vector style phải giữ nhiều loại thông tin; seed trộn content/style từ rất sớm; G chỉ có cơ chế local mixing dựa trên convolution; patch critic và readout chưa được thiết kế chung cho native 32px/64px. Đây là các giới hạn về representation và inductive bias, không phải khẳng định baseline không học được global/local structure.

Nguồn code gốc: [StyleEncoder](https://github.com/ganji15/HiGANplus/blob/5d37de733f79e2a72a79c0f66e1865329ff56b4e/HiGAN%2B/networks/module.py#L65), [Generator](https://github.com/ganji15/HiGANplus/blob/5d37de733f79e2a72a79c0f66e1865329ff56b4e/HiGAN%2B/networks/BigGAN_networks.py#L113), [patch extraction](https://github.com/ganji15/HiGANplus/blob/5d37de733f79e2a72a79c0f66e1865329ff56b4e/HiGAN%2B/networks/utils.py#L308).

### 2.1 Bản đồ khác biệt

| Thành phần | HiGAN+ gốc | ALPHA hiện tại | Cách trình bày contribution |
|---|---|---|---|
| B | Residual CNN; stride đầu cố định; có các intermediate maps | Giữ họ CNN đó; stride/length theo geometry, valid-padding propagation | Nền tảng kế thừa + cải thiện tính đúng đắn/multi-resolution |
| E global | Pooled sequence → MLP → 32-D style | Gated width context → valid mean → normalized MLP → global posterior | Global representation có ngữ cảnh |
| E local | Không có bộ local latent slots độc lập | 7 query slots truy xuất 3 mức feature; 32-D/slot | Contribution 1 |
| Content trước seed | Character embedding ghép style lặp lại | Self-attention + local gated depthwise FFN, rồi content-only seed | Contribution 2 |
| Global style trong G | Một projection rồi chia conditioning theo block | Bốn projection MLP riêng; CBN trong cả bốn blocks | Nâng cấp cơ chế đã có |
| Local style trong G | Không có đường spatial retrieval tương ứng | Feature pixels query 7 local tokens sau GBlock 1 | Contribution 2 |
| Global D | Residual D; readout sum/label-length | Valid-area mean × shared gain; optional width mixer không bật mặc định | Chuẩn hóa thang evidence giữa resolutions |
| Patch D | Residual discriminator trên patch 32×32 | Directional depthwise residual P; spatial logits; crop s=H/2 | Contribution 3 |
| Objective | Bộ loss đa mục tiêu có sẵn | Giữ nhiều mục tiêu; sửa phạm vi global/local, thêm cycle/GRL, scheduled weights | Training design, không nhận toàn bộ loss là mới |
| Native resolution | G gốc định nghĩa 64px | Cùng bốn stages cho 32px/64px | Hỗ trợ hình học thống nhất |

## 3. Tổng quan mô hình hiện tại

{{figure:01_overview|Hình 1. Kiến trúc ALPHA hiện tại. B cung cấp feature chung; E tạo một global code và bảy local tokens. Target text đi vào content-only seed. Global conditioning và local spatial retrieval có hai đường riêng. D/P cùng các teachers cung cấp supervision khi train; inference chỉ cần B/E/G cho reference-guided generation hoặc G và prior cho random generation.}}

Luồng inference chính có thể viết ngắn gọn:

```text
S = E(B(x_ref)) = [g, L1, ..., L7]
C = ContentContext(Embedding(y′))
F0 = Reshape(Linear(C))
F1 = GBlock1(F0; BSSP1(g))
F1′ = SpatialStyleRefinement(F1, L)
x̂ = GBlocks2–4(F1′; BSSP2–4(g))
```

Các GBlocks vẫn là residual convolutional blocks theo nền HiGAN+/BigGAN; không có StarConv, SpiralFC hay texture refinement head trong cấu hình active hiện tại. Attention trong content path và cross-attention trong spatial path không thay thế toàn bộ CNN decoder.

## 4. Contribution 1 — Global–local style representation lấy evidence từ reference

### 4.1 Motivation: tại sao một global vector chưa đủ?

Một reference chứa đồng thời nhiều tín hiệu: độ nghiêng chung, tương phản mực, tỷ lệ nét, độ cong, kiểu nối, cách mở/đóng vòng và bố cục theo chiều cao. Mean pooling có ích để gom đặc điểm chung, nhưng không giữ được toàn bộ tính đa dạng giữa những vùng khác nhau. Khi mọi vị trí trong G chỉ nhận cùng một vector, G phải tự giải mã cả phong cách chung và các biến thể nét từ cùng một bottleneck.

Điều cần bổ sung không phải “mỗi chữ một token được gán nhãn”, mà là **một bộ evidence nhỏ có thể được truy xuất theo nhu cầu**. Một token có thể giữ dấu hiệu về nét vòng hoặc nét kéo dài mà không cần code quyết định đó là chữ `a`, `g`, `/` hay một allograph cụ thể. Các vai trò như vậy là khả năng learned, không phải semantic labels đã được chứng minh.

HWT cho thấy một hướng liên quan: giữ style dưới dạng feature sequence và để query content chọn thông tin style bằng attention. ALPHA không sao chép nguyên HWT; nó giữ global writer code, nén local evidence thành số slots cố định và dùng spatial queries trong CNN G. Đây là quan hệ về nguyên lý, không phải tuyên bố lịch sử rằng mọi thay đổi được lấy trực tiếp từ HWT. [P2]

### 4.2 Backbone B và encoder E phải được phân biệt

B vẫn thuộc họ residual CNN của HiGAN+, với ba feature maps có 64, 128 và 256 channels. B được học cùng W ở pretraining và được giữ frozen trong GAN. **Không nên gọi toàn bộ backbone này là một CNN hoàn toàn mới.** Phần context, query attention và global/local projection mới nằm trong E, được học cùng G.

Đối với native 32px/64px, stride đầu của B được suy ra từ chiều cao: `max(1,H//32)`. Cùng topology và channels nhưng mức giảm kích thước phù hợp hơn: tổng width reduction tương ứng khoảng 8 hoặc 16. Việc truyền chiều dài hợp lệ qua CNN tránh cho batch padding bị tính thành stroke evidence. Đây là xử lý geometry/validity, không phải mask xóa nền bên trong chữ.

Nguồn: [StyleBackbone và length propagation](https://github.com/rquq/HGGAN_test/blob/7eed74c6b2d5286dc7b3e446ec1db3b165322c39/networks/module.py#L11), [StyleEncoder](https://github.com/rquq/HGGAN_test/blob/7eed74c6b2d5286dc7b3e446ec1db3b165322c39/networks/module.py#L301).

### 4.3 Global branch: gom phong cách sau khi đã nhìn ngữ cảnh

Feature sequence cuối của B có 256 channels. Hai context blocks tuần tự dùng depthwise convolution kernel 7 với dilation 1 rồi 2, tương ứng receptive field 19 feature positions. Mỗi block có channel normalization, gated expansion và projection, cộng residual với scale học được. Nó không phải full self-attention dài trên toàn ảnh; ưu tiên width context với chi phí vừa phải.

Sau đó E lấy trung bình chỉ trên chiều dài hợp lệ, chuẩn hóa theo channels, chạy MLP và dự đoán `mu_g`, `logvar_g`. Khi VAE bật, chỉ global code được sample:

```text
g = μg + exp(0.5 logvar_g) · ε,      ε ~ N(0,I)
```

Motivation là tránh gom quá sớm các nét chưa có ngữ cảnh, đồng thời không làm global vector phụ thuộc số cột padding. Normalization giúp scale đầu vào MLP ổn định hơn, nhưng không thể tự khôi phục nét mà reference không có.

### 4.4 Local branch: chọn evidence khác với tạo appearance

{{figure:02_style_encoder|Hình 2. StyleEncoder global–local. Ba feature maps được chiếu về 256 chiều và giữ bốn height rows trước khi tạo local evidence sequence. Position chỉ tham gia keys. Values là visual deviations đã trừ valid-spatial mean riêng từng map. Bảy learned queries truy xuất evidence; đầu projection bias-free tạo 32-D local codes và center chúng theo slots. Global posterior có đường riêng.}}

Với mỗi map k, E pool chiều cao về 4 rows rồi chiếu pointwise về 256 channels. Average pooling và pointwise projection có thể đổi thứ tự mà giữ kết quả toán học; source hiện tại pool trước để giảm lượng phép chiếu. Sequence được chuẩn hóa một lần theo channel:

```text
Uk = LayerNorm(Flatten(Projection(PoolHeight4(Fk))))
Kk = Uk + Position2D(k)
Vk = Uk − Mean_valid_positions(Uk)
```

Các K/V từ ba scales được nối lại. Learned queries Q có shape `[1,7,256]`, được khởi tạo orthogonal với scale tương đương Gaussian cũ. Query chỉ chọn evidence qua 4-head attention; nó không được cộng trực tiếp vào code appearance:

```text
A = MultiHeadAttention(Q, concat(Kk), concat(Vk))
Lj = Wlocal Aj                           # 256 → 32, no bias
Lj ← Lj − Mean_slots(L)
```

Tách K và V giải quyết một vấn đề cụ thể: vị trí và learned query có ích để quyết định “nhìn đâu”, nhưng không nên tự trở thành “nét gì”. Nếu cả PE lẫn query prior chảy trực tiếp vào local output, model có thể tạo representation khá ổn định dù evidence giữa references khác nhau. Thiết kế mới bỏ đường appearance trực tiếp từ query, bỏ reliability gate và dùng projection riêng cho locals.

Valid masks chỉ loại padding ra khỏi attention/mean. **Không có threshold mực, character-ID rule hay special-case cho single digit.** Với reference nghèo thông tin như dấu chấm, model vẫn phải suy luận từ evidence ít ỏi; không có khả năng bảo đảm đoán đúng toàn bộ phong cách writer.

### 4.5 Tại sao center locals? Và giới hạn của lựa chọn này

Centering khuyến khích locals biểu diễn những khác biệt nội bộ thay vì tất cả lặp lại cùng một offset. Cùng contract được áp dụng cho local random prior; global prior không bị đổi. Tuy nhiên, local centering **không chứng minh độc lập thống kê với global**, không bảo đảm các slots có ý nghĩa semantic khác nhau và không ngăn collapse tuyệt đối.

Nó cũng tạo ràng buộc quan trọng: với đúng một local token, `L−mean(L)=0`. Vì vậy, kiến trúc này không có một local branch hữu ích nếu chỉ cấu hình 1 global + 1 local. Với 7 locals, tổng của chúng bằng zero nên còn 6 vector bậc tự do theo trục slots. Đây là trade-off thực sự, cần ghi rõ khi chọn số tokens.

### 4.6 Contribution được nhận đến đâu?

Cách phát biểu hợp lý: **mở rộng style bottleneck của HiGAN+ thành global writer summary và một tập local visual-evidence tokens đa tỉ lệ, đồng thời tách tín hiệu định vị khỏi appearance values**. Không nên nói đã “giải quyết hoàn toàn allographs”, “mỗi slot học một loại chữ” hoặc “orthogonal initialization bảo đảm không collapse”. Những điều đó cần probe/ablation tương ứng.

## 5. Contribution 2 — Content-first generation và spatial style retrieval

### 5.1 Motivation: trộn style sớm có vấn đề gì?

Trong G gốc, style vector được lặp theo text length rồi ghép với từng character embedding trước seed projection. Cách này đơn giản và hiệu quả, nhưng cùng một bước tuyến tính phải vừa dựng content geometry vừa hấp thụ reference style. Khi reference rất ngắn hoặc feature style có scale bất thường, việc can thiệp sớm có thể làm nội dung khó ổn định hơn.

Thay đổi hiện tại phân công rõ hơn: target text tạo structural scaffold; global style điều chỉnh cách render qua các stages; local style bổ sung evidence sau khi feature map đã có vị trí. Đây là một inductive bias, không phải tách rời tuyệt đối content và style: CBN của GBlock 1 vẫn làm spatial query có thông tin global writer.

### 5.2 ContentContext: self-attention và local mixing có hai vai trò

{{figure:03_fusion_generator|Hình 3. ContentContext và hai đường style trong G. Self-attention trộn các target positions; depthwise gated FFN mô hình hóa lân cận theo chiều từ. Global code tạo CBN conditions cho mọi GBlock. Sau GBlock 1, spatial positions query local reference tokens. Hai attention không có cùng input hay cùng nhiệm vụ.}}

Character embedding 120-D đi qua hai residual sublayers:

```text
C1 = C + αcontent · SelfAttention(LN(C), relative_position_bias)
C2 = C1 + αlocal · GatedDepthwiseFFN(LN(C1))
```

Self-attention có 4 heads, tổng attention dimension 128; nó cho phép một ký tự thấy ngữ cảnh xa hơn vài convolution neighbors. Local FFN có hidden width 360, depthwise convolution kernel 5 trên nhánh value, nhân với SiLU gate rồi chiếu về 120 chiều. Nhánh này ưu tiên transitions giữa các vị trí lân cận. Không có lookup quy tắc “chữ nào phải nối với chữ nào”.

Hai scale residual khởi tạo 0.25 nhưng là parameters trực tiếp, không sigmoid ceiling. Con số 0.25 là initialization, **không phải giới hạn maximum strength**. Sau đó content-only linear projection và reshape tạo spatial seed. Việc giữ residual path cho phép model chỉnh mức contextualization thay vì phải thay toàn bộ character embedding ngay từ đầu.

### 5.3 Global path: block-specific không có nghĩa mỗi block nhận style khác writer

Global code g 32-D được đưa qua bốn MLP riêng. Mỗi MLP gồm hai projections với SiLU ở giữa, tạo một conditioning code 32-D cho một GBlock. Writer identity chung vẫn là g; các stages học cách diễn giải nó khác nhau khi dựng hình học, nét và output detail.

HiGAN+ đã có per-block style conditioning. Khác biệt ở đây là **bốn nonlinear projections riêng thay cho một projection rồi split**, và global token được tách khỏi local tokens ở interface. Không nên trình bày BSSP như lần đầu đưa style vào từng block. Nguồn: [BSSP và G wiring](https://github.com/rquq/HGGAN_test/blob/7eed74c6b2d5286dc7b3e446ec1db3b165322c39/networks/BigGAN_networks.py#L40).

### 5.4 Local path: vì sao đưa style vào sau GBlock 1?

Sau block đầu, feature có dạng `[B,256,8,4T]` với T là text canvas length. Mỗi spatial position đã chứa local image context, nên có thể đóng vai trò query. Keys đến từ normalized local codes; values giữ biên độ residual local thay vì normalize chúng rồi trừ global:

```text
Q = Wq LN(Flatten(F1))
K = Wk LN(L)
V = Wv L
F1′ = F1 + αreference · Wo Attention(Q,K,V)
```

Motivation là để những vùng đang hình thành vòng chữ, nét lên hoặc nét xuống có thể chọn evidence khác nhau. Nếu style chỉ được trộn trước seed, query vẫn chỉ là text position; spatial retrieval được đặt sau block đầu để truy vấn trên representation gần với tác vụ render hơn. Đặt quá muộn lại có thể chỉ sửa texture sau khi geometry đã cứng. Lựa chọn block 1 là một trade-off cụ thể, **chưa có ablation chứng minh nó tối ưu hơn mọi vị trí khác**.

Local token order không có slot-ID trong G, nên phép attention bất biến với việc hoán vị locals. Đây là tính chất có ích cho một evidence set, nhưng dẫn tới yêu cầu objective cũng phải tôn trọng tính đối xứng đó. Phần loss ở mục 7 giải thích vì sao không ép E khôi phục đúng thứ tự các random local slots.

Nguồn: [ContentContext](https://github.com/rquq/HGGAN_test/blob/7eed74c6b2d5286dc7b3e446ec1db3b165322c39/networks/fusion.py#L132), [spatial refinement](https://github.com/rquq/HGGAN_test/blob/7eed74c6b2d5286dc7b3e446ec1db3b165322c39/networks/fusion.py#L62), [G forward](https://github.com/rquq/HGGAN_test/blob/7eed74c6b2d5286dc7b3e446ec1db3b165322c39/networks/BigGAN_networks.py#L207).

### 5.5 Allograph learning: diễn giải đúng mục tiêu

Allograph là các cách thể hiện khác nhau của cùng ký tự. Cơ chế hiện tại rộng hơn một “allograph module”: nó truy xuất visual evidence tại spatial features, có thể phục vụ độ cong, nối nét, tương phản hoặc texture. Không có token được gán sẵn cho `a`, `r`, `g`; không có kiểm tra transcript reference để đổi strength.

Model có thể học allographs nếu dữ liệu và image-level losses cung cấp đủ tín hiệu, nhưng không có supervision trực tiếp phân loại allograph. Nếu cần chứng minh contribution đó, nên đo độ nhất quán nét theo writer hoặc làm human study với cùng target/reference, thay vì chỉ dựa vào FID thấp.

### 5.6 Multi-resolution không tạo hai mô hình riêng

G giữ bốn stages và cùng channel schedule `512→256→128→64→64`. Với 64px, stage cuối upsample; với 32px, stage cuối giữ resolution để refine. Output width là `32T` hoặc `16T`. Model không bỏ mất một conditioned stage khi chiều cao nhỏ hơn, và không chọn capacity dựa trên tên dataset hay metric.

Khác biệt nhỏ về parameter count giữa hai heights đến từ shortcut projection ở block có upsampling, không phải một G32 đã bị giảm đi một nửa channels. Đây là hỗ trợ multi-resolution trong cùng họ architecture, không phải scale invariance toán học tuyệt đối. Nguồn: [G_arch](https://github.com/rquq/HGGAN_test/blob/7eed74c6b2d5286dc7b3e446ec1db3b165322c39/networks/BigGAN_networks.py#L16).

## 6. Contribution 3 — Giám sát nét bằng StrokePatchD phù hợp hình học

### 6.1 Motivation: ảnh toàn từ đúng chưa có nghĩa từng nét đúng

Một ảnh có bố cục tổng thể plausible vẫn có thể chứa nét răng cưa, vòng chưa khép, vùng mực mờ hoặc nối nét bị đứt. Global D phải tổng hợp một ảnh variable-width; sau downsampling, những lỗi nhỏ có thể không còn đóng góp đủ mạnh. Ngược lại, chỉ nhìn patch thì không biết từ có hình học toàn cục hợp lý hay không. Vì vậy D và P cần phân công chứ không thay thế nhau.

Ý tưởng kiểm tra cấu trúc cục bộ đã có trong PatchGAN và HiGAN+. Contribution tại đây là **policy lấy crop, directional critic và cách gộp evidence**, không phải ý tưởng local adversarial loss tự thân. [P1, P3]

### 6.2 Crop policy hiện tại — không còn cố định 32×32 ở mọi resolution

{{figure:04_strokepatchd|Hình 4. StrokePatchD. Crop size s=H/2 nên native 32px dùng 16×16, native 64px dùng 32×32. Số crop là clamp(ceil(valid_width/s),4,8). Các start positions được stratify theo trục ngang và luân phiên hai dải dọc. P giữ spatial logits và gộp theo từng từ để từ dài không tự có trọng số lớn hơn chỉ vì có nhiều crops.}}

Trong code HiGAN+ gốc, P kế thừa residual `Discriminator`; `extract_all_patches` lấy overlapping 32×32 windows với stride 8. Do đó baseline đã có local refinement thực sự, không phải một head đơn giản trên global feature. Policy mới chuyển từ sliding-window coverage dày sang số crop ngẫu nhiên có giới hạn, giữ kích thước crop tương đối với height. Đây là thay đổi trade-off coverage/compute và cách weighting, không mặc nhiên là “nhiều supervision hơn”.

Với chiều cao H và valid width W:

```text
s = max(1, floor(H/2))
K = clamp(ceil(W/s), min_crops=4, max_crops=8)
```

Khoảng start positions hợp lệ được chia thành K strata. Mỗi crop có một horizontal start ngẫu nhiên trong stratum của nó; vertical start luân phiên lấy từ dải trên và dưới. Điều này tăng coverage, nhưng **không bảo đảm crops không chồng lấp**: strata phân chia start positions, không phân chia toàn bộ vùng pixel.

Ảnh real, random fake, style-transfer fake và reconstruction fake đều đi qua cùng policy. “Matched” ở đây là **cùng distribution/policy**, không phải cùng tọa độ pixel hoặc patch được ghép nhãn với nhau. Tọa độ được sample độc lập. Từ ngắn hơn crop được điền canonical padding ngoài valid width; không lấy các cột canvas dư làm evidence.

Policy không cần character boxes, ink segmentation hoặc danh sách special characters. Số crop và crop size dựa trên geometry, dùng cùng công thức ở 32px/64px. Đây là một lựa chọn kiến trúc có hyperparameters 4–8, không phải quy tắc ưu ái một tập ký tự benchmark.

### 6.3 P nhìn gì trong mỗi crop?

P có stem 3×3 với 64 channels và ba residual blocks có output channels `128,192,192`. Trong mỗi block, một đường convolution 3×3 học local features; hai depthwise paths `1×5` và `5×1` học thay đổi theo ngang/dọc, sau đó fuse 1×1 và cộng shortcut. Channels có thể trộn trước/sau directional filters, nên module không chỉ nhìn các nét thẳng theo hai hướng.

Downsampling dừng khi một chiều sẽ nhỏ hơn 4 cells. Đầu 1×1 tạo spatial logits, không gom toàn crop thành một vector trước khi đưa ra quyết định. Với crop 16px hay 32px ở cấu hình ba blocks, model giữ được grid 4×4 tại output. Đây là geometry-aware floor, không phải branch riêng cho IAM32.

Nguồn: [StrokePatchBlock/P](https://github.com/rquq/HGGAN_test/blob/7eed74c6b2d5286dc7b3e446ec1db3b165322c39/networks/BigGAN_networks.py#L508), [sampler và aggregation](https://github.com/rquq/HGGAN_test/blob/7eed74c6b2d5286dc7b3e446ec1db3b165322c39/networks/utils.py#L445).

### 6.4 Vì sao thứ tự tính hinge và averaging quan trọng?

Trong D phase, hinge phải được tính **trên từng spatial decision trước khi average**. Nếu average logits trước, một vùng quá dễ có thể bù cho một vùng lỗi mà không bị hinge phạt đúng mức. Sau hinge: mean spatial map → mean các crop trong cùng từ → mean các từ; ba fake branches được average bằng nhau.

Vì mỗi từ có 4–8 crops, việc average per-word ngăn từ dài tự động nhận gradient gấp đôi từ ngắn. Weight `lambda_patch_adv=0.45` chỉ điều chỉnh patch adversarial term trong G objective; D/P objective không đơn giản nhân toàn bộ patch hinge với 0.45.

### 6.5 Global D được giữ để nhìn toàn từ

Global D vẫn thuộc họ residual discriminator. Readout hiện lấy trung bình theo height và valid width, nhân gain chung 32 rồi chạy scalar head. Cách này tránh số feature cells tăng theo height/word length làm đổi thang score. Gain 32 dùng chung cho cả heights, không phải “nếu ảnh 32px thì đổi luật”. Optional WidthContextMixer **không bật trong `gan_iam_32.yml`/`gan_iam_64.yml`, nhưng bật với 4 heads trong `gan_iam_local.yml`**. Vì vậy local profile không chỉ khác batch size; không dùng tốc độ/parameter count của nó thay trực tiếp cho normal profile.

Global D nhận whole-word inputs với augmentation geometry nhẹ trên cả real/fake. Stroke crops chỉ vào P, không bị giả làm “từ hoàn chỉnh” để huấn luyện global D. Đây là bảo toàn semantics của hai critics.

### 6.6 Liên hệ với SpiS-GAN — khác hướng giải quyết

SpiS-GAN đưa Star–Spiral blocks, spiral-modulated discriminator và edge reconstruction vào thiết kế của họ. ALPHA hiện tại không chứa các operators đó. Điểm chung là cùng quan tâm lỗi nét nhỏ sau downsampling; hướng ở đây là style evidence retrieval và một P riêng trên crop, không phải sao chép đường spiral hay Sobel loss. Không có cơ sở nói directional P chắc chắn tốt hơn spiral critic khi chưa ablation. [P4]

## 7. Toàn bộ training objective: loss nào so với cái gì?

### 7.1 Ba ảnh giả, không phải ba generator độc lập

Mỗi G update tạo ba nhóm bằng **cùng một G**:

```text
x̂rand  = G(y′, z)                  # random style, sampled text
x̂style = G(y′, E(B(x_ref)))        # reference style, new text
x̂rec   = G(y,  E(B(x_ref)))        # same reference, original text
```

Reference encoder nhận `org_imgs`; reconstruction target là `style_imgs`, bản đã chuẩn hóa cho training width. Chúng giữ cùng nội dung/writer nhưng không nên mô tả pixel target là raw reference width bất kỳ. CXLoss dùng features nên không yêu cầu output style-transfer có cùng text hay cùng pixel alignment với reference.

{{figure:05_random_flow|Hình 5. Random-generation flow. Random global/local prior và sampled text đi qua G. D/P kiểm tra realism; OCR so với sampled transcript; global re-encoder so với global prior đã stop-gradient. Không có ordered-local reconstruction target trong phiên bản hiện tại.}}

{{figure:06_transfer_flow|Hình 6. Style-transfer flow. Reference cung cấp style; target text khác cung cấp nội dung. D/P kiểm tra realism, R kiểm tra target transcript, W kiểm tra writer identity của reference. Contextual loss so feature reference/transfer không cần pixel alignment; global cycle so code re-encoded với reference global code đã detach.}}

{{figure:07_reconstruction_flow|Hình 7. Reconstruction flow. G nhận transcript gốc và reference style; output được so L1 với ảnh thật đã chuẩn hóa. Reconstruction cũng chịu D/P adversarial. KL đặt lên reference global posterior, không đặt lên pixels và không phải một KL riêng cho từng ảnh giả. CTC không chạy thêm trên reconstruction trong code hiện tại.}}

### 7.2 Bảng loss/target/gradient

| Loss active | Prediction / đối tượng được đánh giá | Target hoặc đối tượng so sánh | Parameters được update trong G phase |
|---|---|---|---|
| Global adversarial | D trên cả ba nhóm fake | Điểm “real” qua `−mean(D(fake))` | G; E với reference branches |
| Patch adversarial | P trên crop của cả ba nhóm fake | Điểm “real” qua local critic | G; E với reference branches |
| Random CTC | R(x̂rand) | Sampled transcript y′ | G |
| Style CTC | R(x̂style) | Sampled transcript y′ | G và E |
| Info / random style reconstruction | E_global(B(x̂rand)) | stop-gradient(z_global) | G và global path của E |
| Style cycle | E_global(B(x̂style)) | stop-gradient(g_reference) | G và E |
| Pixel reconstruction | x̂rec trên valid pixels | x đã chuẩn hóa width | G và E |
| Writer CE | W(B(x̂style)) | Writer w của reference | G và E; W/B frozen |
| Contextual | B features của x̂style | B features reference đã detach | G và E; B frozen |
| Content-adversarial | Probe(global reference code) | Bag-of-characters của transcript reference | Probe học phân loại; E nhận gradient đảo dấu qua GRL |
| KL | μg, logvar_g của reference | Standard normal N(0,I) | Global posterior của E |
| Real/fake hinge, R1 | D/P real-vs-fake; R1 trên real input của D | Real/fake sign, gradient penalty | Chỉ D/P trong critic phase |

“E nhận gradient” không có nghĩa mọi parameter của E nhận gradient trực tiếp từ mọi loss. Ví dụ fake global-only re-encoding bỏ qua local retrieval; KL không tác động trực tiếp vào local posterior vì không có local Gaussian posterior. Local evidence được học qua các image-generation objectives của reference branches.

### 7.3 Công thức tổng hợp và các weights đang dùng

Đặt `Aglobal = −(D(x̂rand)+D(x̂style)+D(x̂rec))/3`; `Apatch` có cấu trúc tương tự sau per-word aggregation. G objective là:

```text
LG = Aglobal + 0.45 Apatch
   + λctc_rand(t) CTC(R(x̂rand), y′)
   + λctc_style(t) CTC(R(x̂style), y′)
   + 1.0 Linfo + 1.0 Lcycle + 0.02 Lcontent_adv
   + 0.5 Lwriter + λrec(t) Lpixel + 0.1 Lcontext + 0.1 LKL
```

| Weight / schedule | Giá trị config hiện tại | Motivation; không phải kết luận metric |
|---|---|---|
| Random CTC | 1.0 → 0.4, epochs 10–20 | Giữ readability; giảm áp lực canonicalization về sau |
| Style CTC | 0.5 → 0.20, epochs 8–28 | Nhẹ hơn random path để không ép nét writer thành dạng OCR ưa thích |
| Pixel reconstruction | 10.0 → 2.5, epochs 24–42 | Học alignment sớm, giảm ưu tiên pixel averaging về sau |
| Info / style cycle | 1.0 / 1.0; global mean L1 chia 8 | Giữ global conditioning và transfer consistency |
| Content adversarial | 0.02, BCE bag-of-characters qua GRL, chia 8 | Hạn chế global code mang transcript reference |
| Global KL | 0.1 × mean KL / 8 | Khớp prior mà không ép local evidence thành Gaussian |
| Writer / contextual | 0.5 / 0.1 | Giữ writer cues và non-aligned feature statistics |
| Global-D R1 | λ=0.01, mỗi 32 critic iterations, lazy-scaled | Smooth real-input critic gradients; không phải loss làm nét ảnh trực tiếp |

Đây là giá trị **code/config hiện tại**, không được giả định áp dụng từ epoch đầu đến cuối mọi checkpoint lịch sử đã resume.

### 7.4 Vì sao bỏ ordered-local latent reconstruction?

G xem locals như một set không thứ tự. Nếu hoán vị các local slots, attention và ảnh output gần như không đổi. Nhưng nếu Info loss bắt E khôi phục đúng ordered local vectors của random input, cùng một ảnh có thể bị yêu cầu trả về nhiều thứ tự target khác nhau. Đây là non-identifiability của objective, không phải chỉ thiếu capacity.

Phiên bản hiện tại chỉ reconstruct global code. Các local tokens không bị ép phải có thứ tự semantic cố định mà G không sử dụng. Việc chia global loss cho 8 giữ contribution của global tương ứng với all-token mean cũ; nếu bỏ hệ số mà giữ lambda, global supervision sẽ tăng tám lần so với contribution global trước đó. MAIN 5-token là một experiment khác và không thuộc cấu hình ALPHA của báo cáo này.

Đối với content-adversarial, dự án dùng gradient reversal theo nguyên lý adversarial representation learning; ở đây “nuisance” cần hạn chế là content của reference, không phải domain label như DANN. GRL là cơ chế kế thừa ý tưởng đã có, không phải layer mới do dự án phát minh. [P7]

### 7.5 Cái gì không còn là loss/contribution active?

Không có frequency-distribution loss active, texture-refinement head active hay loss riêng gán cho `a`, `b`, `/` và single digits. Gram helper còn trong utility không có nghĩa Gram loss đang tham gia LG; cần nhìn call site và config. Contextual loss vẫn bật ở weight 0.1 và là thành phần kế thừa, không nên đổi tên nó thành contribution mới.

HiGAN+ dùng contextual matching vì reference và output khác text không thể so L1 theo pixel. Hiện tại loss được triển khai valid-length-aware, trừ mean theo từng sample và giới hạn tối đa 256 feature positions trước matching để tránh chi phí quadratic quá lớn. Đây là một adaptation của contextual loss và một cải thiện implementation, không phải công thức loss hoàn toàn mới. [P6]

Nguồn objective: [G/D training loop](https://github.com/rquq/HGGAN_test/blob/7eed74c6b2d5286dc7b3e446ec1db3b165322c39/networks/model.py#L2090), [global objectives](https://github.com/rquq/HGGAN_test/blob/7eed74c6b2d5286dc7b3e446ec1db3b165322c39/networks/model.py#L2350), [loss implementations](https://github.com/rquq/HGGAN_test/blob/7eed74c6b2d5286dc7b3e446ec1db3b165322c39/networks/loss.py#L42).

## 8. Các cải tiến hỗ trợ: cần báo cáo nhưng không nên thổi thành contribution chính

### 8.1 Frozen teachers và gradient đi qua ảnh

W/B được giữ ở eval mode trong GAN; parameters không được optimizer update. R cũng eval, riêng phần cuDNN RNN giữ training flag cần thiết cho backward qua input. “Frozen teacher” không có nghĩa dùng no-grad khi chấm ảnh fake: gradient vẫn phải đi qua R/W/B để tới generated pixels, G và E.

Native W+B/R cho 32px và 64px được pretrained riêng theo resolution/data contract. Cùng tên loss nhưng teacher khác chất lượng có thể đổi training dynamics. Vì vậy đây cũng là một biến cần kiểm soát khi muốn chứng minh một module GAN hiệu quả.

Nguồn: [GAN set_mode override](https://github.com/rquq/HGGAN_test/blob/7eed74c6b2d5286dc7b3e446ec1db3b165322c39/networks/model.py#L968).

### 8.1.1 W thay mean pooling bằng attentive statistics pooling

Writer head trong current config dùng `attentive_stats`, khác với uniform mean pooling của W gốc. Một mạng nhỏ học trọng số theo width positions hợp lệ; W lấy cả weighted mean và weighted standard deviation của 256-D features, ghép thành 512-D rồi chiếu về 256-D trước classifier. Motivation là hai ảnh có cùng mean feature vẫn có thể khác mức biến thiên nét/spacing; std cung cấp thông tin mà mean không giữ. Attention có thể giảm ảnh hưởng vùng ít thông tin, nhưng không có bảo đảm tự tìm đúng mọi stroke.

Đây là thay đổi của **writer teacher/pooling**, không phải local-token attention trong E. W không có bảy slots và không trực tiếp sinh ảnh. Khi chấm transfer image, gradient qua W/B hướng G/E giữ writer cues. Để so architecture GAN công bằng, cần giữ nguyên cùng bộ W+B teachers ở các ablations; teacher tốt hơn có thể là một phần nguyên nhân cải thiện toàn hệ thống nhưng không được gán nhầm cho fusion.

### 8.1.2 R và pretraining contract

R vẫn là residual CNN + BiLSTM/CTC theo họ HiGAN+, không bị đổi thành Transformer OCR. Thay đổi chính là stride/length support cho native 32/64, alphabet 81 classes, API log-probability rõ ràng và pipeline polarity nhất quán. Generated tensors được đưa trực tiếp vào R trong GAN, không invert lần nữa giữa G và teacher. Padding background theo convention nội bộ không đồng nghĩa màu hiển thị của PNG export.

Current pretrain configs đặt 100 epochs, eval mỗi 10 epochs và EMA teacher có thể bật/tắt. Writer pretraining dùng mixture raw/style-normalized inputs, balancing theo writer và attentive pooling; OCR có augmentation label-preserving. Các lựa chọn này phục vụ robust teacher features, nhưng báo cáo không có ablation để nói riêng EMA hay augmentation đã giảm KID bao nhiêu. W validation trên `trnval` đo closed-set training-writer classification; không nên dùng WIER thấp ở đó làm bằng chứng nhận diện unseen test writers.

### 8.2 Rare characters mà không hardcode từng case

Sampler tăng xác suất của từ thật trong corpus có ký tự hiếm, dùng inverse-square-root character frequency. Nó không tạo chuỗi punctuation/digit ngẫu nhiên không xuất hiện trong training corpus, cũng không bật một module khác khi reference là dấu chấm. Alphabet hiện có 81 classes bao gồm blank và `*`; unknown symbols báo lỗi thay vì âm thầm map sang ký tự khác.

VATr++ là đối chiếu phù hợp cho motivation: distribution của text và bias của auxiliary networks ảnh hưởng long-tail glyphs; chuẩn hóa evaluation cũng quan trọng. Dự án chọn oversample từ có thật, khác với augmentation thay ký tự của VATr++. Không nên viết rằng đã cài nguyên thuật toán VATr++ nếu chỉ cùng xử lý vấn đề imbalance. [P5]

### 8.3 Padding correctness khác với foreground masking

Các valid-length masks còn tồn tại ở CNN features, attention, reconstruction và output tail. Chúng loại phần batch canvas không thuộc ảnh; không cắt nền giấy trong valid region, không threshold texture và không ép foreground thành binary. Xóa những masks này có thể làm padding chi phối style/pooling/loss. Vì vậy câu “đã bỏ masking” chỉ nên dùng nếu nói đúng loại foreground/background masking đã tắt, không phải toàn bộ validity handling.

### 8.4 Efficiency và reproducibility

Ba fake branches được batch chung khi forward G. Real B features được cache trong batch; fake global-only re-encodings không chạy local retrieval không dùng đến. Writer/context losses dùng lại feature pass của transfer. EMA giữ snapshot G/E và đồng bộ buffers. Checkpoint lưu optimizer, scheduler, RNG, progress và eval history; CSV ngoài run folder có thể được tái tạo từ checkpoint.

Đây là các cải tiến engineering giúp training dễ kiểm tra và ít làm việc thừa hơn. Không có benchmark thời gian kiểm soát trong lần báo cáo này, nên không quy đổi chúng thành tuyên bố “nhanh hơn X%”. Toàn bộ profile đang mô tả là FP32.

## 9. Kết quả thực nghiệm đã lưu: điều gì thực sự được xác nhận?

### 9.1 Nguồn bằng chứng

Đã đọc `.pth` trên CPU, kiểm tra các hàng CSV nhúng với CSV phân tích cũ; cả chín checkpoint được kiểm tra có lịch sử khớp chính xác. Không lấy tên file làm bằng chứng duy nhất, không ghép best FID của một epoch với best KID/HWD của epoch khác. Bảng dưới dùng **ba metrics cùng một hàng epoch**. KID trong evaluator này là polynomial MMD trung bình **nhân 100**.

| Snapshot | Checkpoint / epoch | FID ↓ | KID ×100 ↓ | HWD ↓ |
|---|---|---:|---:|---:|
| ALPHA 07/10 | best / 46 | 3.261411 | 0.103138 | 0.123412 |
| ALPHA 07/10 | last / 48 | 3.318018 | 0.114214 | 0.126290 |
| ALPHA 14/10 | best FID / 74 | **2.446330** | 0.063558 | **0.093104** |
| ALPHA 14/10 | alternate best / 61 | 2.505184 | **0.063397** | 0.113436 |
| ALPHA 14/10 | last / 98 | 2.821970 | 0.094728 | 0.103072 |
| MAIN 14/10, đối chiếu lịch sử | best / 87 | 2.759759 | 0.091686 | 0.078269 |

**Tất cả các hàng trên là 32px.** Không dùng chúng làm bằng chứng trực tiếp rằng 32px tốt hơn/kém hơn 64px. MAIN trong bảng là architecture và fine-tune config cũ, không phải MAIN đã đồng bộ code/token experiment ngày 14/10.

Từ hai snapshot ALPHA best, FID thấp hơn khoảng 25.0%, KID thấp hơn khoảng 38.4% và HWD thấp hơn khoảng 24.6%. Đây chỉ là chênh lệch giữa **hai kết quả lịch sử khác architecture và training budget**, không phải ước lượng causal effect của spatial fusion hoặc encoder. Run 07/10 còn có StarConv/ReferenceStyleAttention; 14/10 có ContentContext/SpatialStyleRefinement và train lâu hơn.

So last98 với best74 trong cùng lịch sử 14/10, FID tăng khoảng 15.36%, KID tăng 49.04%, HWD tăng 10.71%. Điều này cho thấy last không phải checkpoint tốt nhất và late training có regression; không có nghĩa model collapse toàn cục hay LR là nguyên nhân duy nhất. Chênh lệch KID giữa epoch61 và74 chỉ khoảng 0.000160, không có repeat/error bars để khẳng định epoch61 tốt hơn có ý nghĩa thống kê.

### 9.2 Có thể so với HiGAN+ hoặc SpiS-GAN không?

| Nguồn | Model / resolution | FID | KID | HWD | Giá trị sử dụng |
|---|---|---:|---:|---:|---|
| SpiS-GAN paper, IAM table | HiGAN+ / 64px | 10.17 | khoảng 0.44–0.45 tùy bảng | 0.33 | Mốc từ publication, không phải paired local benchmark |
| SpiS-GAN paper, IAM table | SpiS-GAN / 32px | 4.37 | 0.06 | 0.51 | Mốc từ publication; protocol cần đối chiếu |
| Local native32 baseline benchmark | SpiS-GAN / 32px | 4.581947 | 0.086179 | 0.171112 | Kết quả script benchmark cũ, không rerun lần này |
| Local native32 baseline benchmark | FW-GAN / 32px | 6.308192 | 0.189474 | 0.219752 | Cùng đợt external-model benchmark |
| ALPHA embedded eval | ALPHA / 32px, epoch74 | 2.446330 | 0.063558 | 0.093104 | Kết quả in-training của checkpoint lịch sử |

Các con số có vẻ thuận lợi cho ALPHA, nhưng chưa đủ để viết “đã vượt SOTA trong cùng protocol”. Benchmark external-model cũ có 25,766 examples; protocol x64 cũ còn resize outputs từ native32 và có polarity bridge. Audit native32 cũng thấy polarity/generation path chưa khớp hoàn toàn: adapter giữ nguồn native không convert, còn local GAN có source-polarity normalization; external generator nối sampled noise vào style, local path dùng trực tiếp encoder code; HWD batch là 64 so với local 32. Local history dùng EMA G/E, trong khi external checkpoint không có EMA key riêng để xác nhận tương đương. Số mẫu/dataset identity của từng hàng lịch sử chưa được mã hóa đủ để chứng minh cùng row set. Vì vậy cùng một implementation FID/KID không tự động có nghĩa toàn bộ protocol giống nhau. Đặc biệt HWD paper 0.51 và local 0.171 không nên bị coi là cùng một phép đo chỉ vì cùng tên metric. [P4, P5]

Không tìm được trong phạm vi bằng chứng này một paired HiGAN+ gốc/native32 run với cùng teachers, cùng evaluator và cùng generation list để tính phần trăm cải thiện đáng tin cậy. Baseline gốc vốn có G64; port native32 cũng cần được ghi rõ là port, không giả làm kết quả nguyên bản. Bảng publication chỉ là context.

### 9.3 Điểm thấp nhưng local branch có thật sự hoạt động không?

Phân tích lưu ngày 14/10 trước lần sửa mới nhất đã probe checkpoint best74 trên 21 references. Mean cosine giữa normalized local slots khoảng 0.96846; normalized attention entropy khoảng 0.98846. Đổi locals sang writer khác trong khi giữ global chỉ đổi image L1 khoảng 0.000018, trong khi đổi global đổi L1 khoảng 0.34922.

Đây là dấu hiệu local representation gần collapse trong diagnostic đó, dù aggregate FID/KID tốt. Spatial branch cũ còn dùng normalized-local trừ normalized-global, nên vẫn có thể hoạt động như một global correction thay vì local evidence retrieval. Bảy slots tồn tại trong tensor **không chứng minh** bảy loại handwriting evidence được dùng.

Từ đây mới có motivation cho code hiện tại: bỏ ordered-local latent target, không cộng query prior vào local appearance, tách K/V, dùng projection local bias-free và bỏ nested reliability gates. Cơ chế mới giải quyết mismatch hợp lý về toán học và đường thông tin; **chưa có FID/KID/HWD sau controlled retraining để xác nhận kết quả**.

## 10. Cơ sở nghiên cứu: bảy bài báo và vai trò thực sự

Bốn bài đầu/đối chiếu chính đã được đọc phần phương pháp từ PDF có sẵn: HiGAN+, HWT, SpiS-GAN, VATr++. Ngoài ra đã kiểm tra full-text/nguồn chính thức cho PatchGAN/pix2pix và nguồn gốc contextual/GRL. Các liên hệ dưới đây là **cơ sở khoa học để giải thích design**, không tự nhận đó là lịch sử quyết định thực tế của mọi commit.

| Paper | Điều dùng để lý giải | Không được suy diễn |
|---|---|---|
| [P1] HiGAN+ | Baseline disentanglement, CBN, writer/OCR/context, patch refinement | Không nhận những thành phần này là đóng góp mới |
| [P2] HWT | Style sequence và attention có thể giữ/retrieve global/local cues | Không nói ALPHA là HWT hoặc đã chứng minh superiority |
| [P3] pix2pix/PatchGAN | Local adversarial constraints bổ sung pixel/global objectives | Không nói patch loss lần đầu xuất hiện ở dự án |
| [P4] SpiS-GAN | Đối chiếu cách xử lý stroke geometry và mất chi tiết do downsampling | Không gán Star/Spiral/Sobel cho code hiện tại |
| [P5] VATr++ | Long-tail input distribution, auxiliary bias, evaluation protocol | Không gọi corpus oversampling là nguyên thuật toán thay ký tự của paper |
| [P6] Contextual Loss | So features không cần spatial alignment | Không gọi nó là KID loss hay một loss mới của dự án |
| [P7] DANN | Gradient reversal để làm representation bớt chứa nuisance information | Không khẳng định domain invariance và content disentanglement là cùng bài toán |

## 11. Contribution statements có thể dùng khi báo cáo

**C1 — Representation.** “Chúng tôi mở rộng latent style của HiGAN+ thành một global writer code và một tập local visual-evidence tokens đa tỉ lệ. Các local queries chọn evidence qua keys có vị trí, trong khi values chỉ mang visual deviations; nhờ đó giảm đường appearance không phụ thuộc reference.”

**C2 — Generation.** “Chúng tôi phân tách content construction và style rendering: target text được contextualize trước khi dựng seed, global code điều kiện hóa từng GBlock, còn local evidence được truy xuất bởi spatial features sau stage đầu. Thiết kế không cần character-specific allograph rules.”

**C3 — Local supervision / geometry.** “Chúng tôi thiết kế StrokePatchD với directional residual filters và spatial decisions trên các half-height crops được stratify theo valid word geometry. Per-word aggregation và shared multi-resolution topology giữ semantics giám sát nhất quán giữa native 32px/64px.”

Nên đi kèm một câu giới hạn: “Đây là các cải tiến thiết kế so với HiGAN+; hiệu quả riêng của từng thành phần cần ablation với cùng initialization, teachers, training budget và evaluation protocol.” Không dùng các từ “first”, “novel attention” hoặc “guaranteed disentanglement” khi chưa có literature/experiment đủ mạnh.

### 11.1 Nếu thầy hỏi “tại sao không chỉ dùng Transformer fusion khác?”

Mục tiêu không phải chứng minh attention type này ưu việt tuyệt đối. Dự án giữ decoder CNN variable-width của HiGAN+ và đặt attention vào hai điểm có nhiệm vụ rõ: target sequence context và spatial evidence retrieval. Full Transformer decoder là một đối chứng hợp lệ, nhưng có capacity/compute khác và không tự giải quyết ordered-local target mismatch. Cần so dưới cùng budget trước khi kết luận.

### 11.2 “Tại sao local tokens mà không giữ toàn bộ feature map?”

Giữ toàn bộ map bảo toàn nhiều thông tin hơn nhưng tăng số keys/values theo reference width, chi phí spatial cross-attention và nguy cơ mang content reference vào G. Bộ slots cố định là bottleneck có chủ đích để nén evidence. Trade-off là có thể mất style hiếm hoặc collapse; lựa chọn bảy slots là hyperparameter, không phải con số có bảo đảm lý thuyết.

### 11.3 “Tại sao cần P nếu đã có D và OCR?”

D nhìn realism toàn từ; OCR ưu tiên transcript và có thể chấp nhận ảnh đọc đúng nhưng nét không giống writer; P ưu tiên local plausibility của strokes. Chúng cung cấp tiêu chí khác nhau. Tuy nhiên, P quá mạnh cũng có thể làm background/texture lấn át content, nên cần per-word weighting và ablation weight/crop policy thay vì thêm critic không giới hạn.

## 12. Ablation cần có để biến design contribution thành bằng chứng

| Câu hỏi | So sánh công bằng | Điều cần đo |
|---|---|---|
| Local evidence có cần không? | Global-only với đường local được bypass đúng, so 1+4, 1+7; giữ effective loss normalization | FID/KID/HWD, reference swap sensitivity, diversity |
| K/V separation có ích không? | Cùng G, cùng slots; PE ở K-only so PE ở K/V; không đổi loss khác | Giữa-writer local variation, attention entropy, image changes |
| Spatial insertion hơn early fusion? | Giữ E/B/teachers và capacity gần nhau, chỉ đổi vị trí/cách fusion | Image metrics, short-reference tests, latency |
| P mới có ích không? | P gốc so directional P, cùng sampled crops; sau đó mới đổi crop policy | KID/HWD, jagged strokes, per-crop errors |
| Multi-resolution công bằng chưa? | Cùng topology/weights schedule và compute budget; native data ở từng height | Cả chất lượng lẫn number of G updates |
| Loss mismatch fix có ích không? | Ordered-local target so global-only target với scale tương đương | Metric trajectories và local-collapse probes |

Tối thiểu nên có nhiều seeds và cùng một danh sách evaluation references/targets. Train từ checkpoint khác nhau không phải controlled on/off comparison. Chọn best theo FID không đồng nghĩa tối ưu mọi metric; cần báo cả KID/HWD cùng epoch và last để thấy trajectory.

Reference swap test cũng cần cẩn thận: cùng global, đổi local writer; cùng local, đổi global; dùng nhiều writers và cùng targets. Một local branch làm ảnh thay đổi lớn chưa chứng minh thay đổi đúng style. Có thể bổ sung blinded preference study hoặc handwriting-aware feature similarity trên các pairs được kiểm soát.

## 13. Giới hạn và những điều không nên claim

1. **Một reference nhỏ không xác định duy nhất writer style.** Dấu `/` hoặc `.` không chứa đủ evidence cho mọi vòng chữ/ligature. Learned processing có thể generalize tốt hơn hardcoded cases nhưng không tạo ra thông tin đã mất.
2. **Style slots chưa được gán semantic labels.** Không có bằng chứng slot 1 là loops và slot 2 là descenders. Muốn nói điều đó cần visualization/intervention có hệ thống.
3. **Current E chưa có full retraining score.** FID 2.4463 là evidence cho checkpoint trước sửa, không phải chứng nhận cho K-only positional evidence mới.
4. **Metrics không thay thế qualitative analysis.** FID/KID dùng feature statistics; low scores không bảo đảm spelling, allographs, background match hay writer identity đều đúng ở từng mẫu.
5. **P không đảm bảo nền đồng nhất.** Nó chấm cả crop, bao gồm background trong valid image. Có thể học paper texture hữu ích hoặc artifacts; không có background-segmentation loss active.
6. **Native 32/64 không hoàn toàn tương đương dữ liệu.** Cùng topology không xóa được khác biệt resolution, preprocessing hoặc corruption. Không thay dataset benchmark để lấy điểm tốt hơn.
7. **Parameter count không suy ra tốc độ.** Kernel launches, attention, teacher passes, widths và R1 đều ảnh hưởng wall time. Đợt viết này không đo throughput mới.

## 14. Kết luận

Điểm mạnh nhất để trình bày không phải một danh sách dài các blocks và losses, mà là **một câu chuyện thông tin xuyên suốt model**: giữ global style đủ ổn định, nén local evidence mà không cài character-specific rules, dùng target content dựng structure, truy xuất local evidence tại spatial representation, rồi kiểm tra realism ở cả whole-word lẫn stroke scale.

So với HiGAN+, ALPHA hiện tại có những thay đổi thực sự ở E, content/style interface trong G, local critic và resolution handling. Nhiều thành phần nền — residual CNN, CBN, OCR/writer teachers, contextual matching và dual critics — vẫn được kế thừa và phải ghi nhận đúng.

Kết quả lưu trên native32 đã đạt FID khoảng 2.45 và KID khoảng 0.064; đó là một cột mốc đáng báo cáo, đồng thời local-collapse probe nhắc rằng điểm tốt chưa đủ chứng minh representation đúng ý định. Contribution có sức thuyết phục nhất khi **motivation, mechanism, code evidence và controlled ablation** đều nối với nhau, thay vì chỉ đặt một tên module mới cạnh một con số tốt.

## Phụ lục A. Hyperparameters và parameter counts

| Profile | G/D/P LR | Critic:G | Train batch | Epochs | LR decay start/duration/floor |
|---|---|---:|---:|---:|---|
| gan_iam_32 | 2e-4 / 2e-4 / 2e-4 | 2:1 | 8 | 100 | 36 / 48 / 0.15 |
| gan_iam_64 | 2e-4 / 2e-4 / 2e-4 | 2:1 | 8 | 100 | 36 / 48 / 0.15 |
| gan_iam_local | 2e-4 / 2e-4 / 2e-4 | 2:1 | 2 | 84 | 36 / 48 / 0.15 |

Adam betas 0.5/0.999, gradient clip 5, EMA beta 0.9998, alphabet 81 classes, style_dim 32, total tokens 8. FID/KID/HWD bật; CER/WER tắt trong current normal configs. Critic:G=2:1 nghĩa là D/P được update hai batches cho mỗi G/E update, không phải tăng batch G lên gấp đôi. Local profile bật thêm WidthContextMixer trong D; normal32/64 không bật.

| Module | 32px parameters | 64px parameters | Vai trò |
|---|---:|---:|---|
| G | 3,942,493 | 3,946,653 | Train trong GAN |
| E hiện tại | 1,332,913 | 1,332,913 | Train trong GAN |
| B | 1,617,328 | 1,617,328 | Frozen teacher/features |
| W | 317,780 | 317,780 | Frozen writer head |
| R | 2,428,673 | 2,428,673 | Frozen OCR teacher |
| D | 2,366,145 | 2,366,145 | Train trong GAN |
| P | 794,945 | 794,945 | Train trong GAN |

Các counts là normal32/64, lấy từ constructor snapshots/verification artifacts ngày 14/10; không đo lại GPU trong báo cáo này. Không áp D count này cho local profile bật WidthContextMixer. E cũ có 1,391,666 parameters, E mới giảm 58,753 (~4.22%). Không dùng số giảm đó để suy ra wall-time speedup tương ứng. Deployment reference-guided cần tính cả B+E+G, không chỉ G+E; teachers R/W và critics chỉ cần khi training/evaluation liên quan.

## Phụ lục B. Evidence và source map

| Nội dung | File / vị trí đã kiểm tra |
|---|---|
| Global/style context, B, E | `HGGAN_test/alpha/networks/module.py:105,167,301,438,455,548` |
| Content self-attention, depthwise FFN, spatial retrieval | `networks/fusion.py:14,62,132,166` |
| Shared 32/64 G, BSSP, G forward | `networks/BigGAN_networks.py:16,40,207` |
| Global D và P | `networks/BigGAN_networks.py:385,467,508,542` |
| Prior centering, rare words, stroke crop/aggregation | `networks/utils.py:244,263,445,498` |
| Frozen teacher modes | `networks/model.py:968` |
| GAN fake branches, objectives | `networks/model.py:2090,2230,2350,2500` |
| Contextual/global loss helpers | `networks/loss.py:42,236,260` |
| KID scale | `metric/val_metrics.py:458` |
| Checkpoint report cũ | `test_artifacts/checkpoint_analysis_14_10/report.md` |
| CSV đã đối chiếu | Các CSV trong `test_artifacts/checkpoint_analysis_7_10` và `checkpoint_analysis_14_10` |
| Bằng chứng mới trong report bundle | `evidence/metrics_evidence_summary.json`, `evidence/checkpoint_metrics_extract.csv`, `evidence/extract_checkpoint_metadata.py` |

Mọi đường tương đối `networks/...` trong bảng này được tính từ `/home/quq/machineLearning/HTG/HGGAN_test/alpha`. Các paper text extracts chỉ là bản đọc nội bộ của PDFs có sẵn, không phải tài liệu do dự án viết.

## Tài liệu tham khảo

**[P1]** Ji Gan, Weiqiang Wang, Jiaxu Leng, Xinbo Gao. *HiGAN+: Handwriting Imitation GAN with Disentangled Representations*. ACM Transactions on Graphics, 42(1), article 11; publication 2022, issue metadata có thể ghi 2023. [DOI: 10.1145/3550070](https://doi.org/10.1145/3550070). Đã đối chiếu PDF 17 trang và code tác giả: [ganji15/HiGANplus](https://github.com/ganji15/HiGANplus).

**[P2]** Ankan Kumar Bhunia, Salman Khan, Hisham Cholakkal, Rao Muhammad Anwer, Fahad Shahbaz Khan, Mubarak Shah. *Handwriting Transformers*. ICCV 2021, pp.1086–1094. [CVF paper](https://openaccess.thecvf.com/content/ICCV2021/html/Bhunia_Handwriting_Transformers_ICCV_2021_paper.html), [arXiv:2104.03964](https://arxiv.org/abs/2104.03964). Đã đọc phần method và đối chiếu PDF kèm supplementary.

**[P3]** Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, Alexei A. Efros. *Image-to-Image Translation with Conditional Adversarial Networks*. CVPR 2017. [CVF PDF](https://openaccess.thecvf.com/content_cvpr_2017/papers/Isola_Image-To-Image_Translation_With_CVPR_2017_paper.pdf), [arXiv:1611.07004](https://arxiv.org/abs/1611.07004). Dùng để đối chiếu nguyên lý PatchGAN, không coi implementation của HiGAN+ là đúng cùng kiến trúc pix2pix.

**[P4]** Nguyen Duy Hieu, Dang Hoai Nam, Pham Hoang Giap, Quang Huu Hieu, Vo Nguyen Le Duy. *SpiS-GAN: Spiral-Modulated Handwriting Synthesis with Star Operation*. arXiv preprint, 2026. [arXiv:2607.06949](https://arxiv.org/abs/2607.06949), [author repository](https://github.com/DAIR-Group/SpiS-GAN). Đã đối chiếu PDF/full text phần phương pháp và IAM tables.

**[P5]** Bram Vanherle, Vittorio Pippi, Silvia Cascianelli, Nick Michiels, Frank Van Reeth, Rita Cucchiara. *VATr++: Choose Your Words Wisely for Handwritten Text Generation*. 2024. [arXiv:2402.10798](https://arxiv.org/abs/2402.10798). Đã đọc phần input preparation, auxiliary regularization và evaluation protocol trong PDF.

**[P6]** Roey Mechrez, Itamar Talmi, Lihi Zelnik-Manor. *The Contextual Loss for Image Transformation with Non-Aligned Data*. ECCV 2018. [arXiv:1803.02077](https://arxiv.org/abs/1803.02077), [author code](https://github.com/roimehrez/contextualLoss). Dùng để xác nhận mục tiêu non-aligned feature matching và nguồn gốc loss.

**[P7]** Yaroslav Ganin, Evgeniya Ustinova, Hana Ajakan, Pascal Germain, Hugo Larochelle, François Laviolette, Mario Marchand, Victor Lempitsky. *Domain-Adversarial Training of Neural Networks*. JMLR 17(59), 2016. [JMLR](https://jmlr.org/papers/v17/15-239.html), [arXiv:1505.07818](https://arxiv.org/abs/1505.07818). Dùng cho cơ chế gradient reversal; task của ALPHA là content nuisance, không phải domain adaptation nguyên bản.
