"""Extract existing research PDFs for the report audit (bundled Python + pypdf)."""
from pathlib import Path
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parent
LIBRARY = Path('E:/CODING/MachineLearning/papers/HTG')
for name in ('higanplus.pdf', '2021_Handwriting_Transformers.pdf',
             'spis-gan.pdf', '2024_VATr_PlusPlus.pdf'):
    reader = PdfReader(LIBRARY / name)
    text = '\n\n'.join(f'--- PAGE {i + 1} ---\n{page.extract_text()}'
                        for i, page in enumerate(reader.pages))
    destination = ROOT / 'evidence' / 'papers' / (Path(name).stem + '.txt')
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text, encoding='utf-8')
    print(f'{name}: {len(reader.pages)} pages, {len(text)} characters extracted')
