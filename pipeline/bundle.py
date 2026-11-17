"""One path-preserving resume archive; .zip.keep avoids uploader auto-unpacking."""
import argparse
import os
from pathlib import Path
import stat
import tempfile
import zipfile


def export_bundle(output, archive):
    output, archive = Path(output).resolve(), Path(archive).absolute()
    if archive.is_relative_to(output):
        raise ValueError('Resume archive must be outside the output folder')
    if not (output/'prepared/identity.json').is_file():
        raise ValueError('Output has no prepared cohort')
    archive.parent.mkdir(parents=True,exist_ok=True)
    fd,temporary=tempfile.mkstemp(prefix='.resume-',dir=archive.parent);os.close(fd)
    try:
        with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
            for directory, dirs, files in os.walk(output,followlinks=False):
                directory=Path(directory);relative=directory.relative_to(output)
                # Native checkpoint-N state reconstructs these transport aliases.
                omit_aliases=len(relative.parts)==3 and relative.parts[0]=='ocr'
                dirs[:]=[d for d in dirs if not (directory/d).is_symlink()
                         and not d.startswith('.restored-alias-')
                         and not (omit_aliases and d in ('best','last'))]
                for name in sorted(files):
                    path=directory/name
                    if path.is_symlink() or path.suffix=='.pid':continue
                    z.write(path,Path('ocr_benchmark_outputs')/path.relative_to(output))
        os.replace(temporary,archive)
        return archive
    finally:
        if os.path.exists(temporary):os.unlink(temporary)


def restore_archive(archive,destination):
    from .preflight import restore_outputs
    destination=Path(destination).absolute()
    destination.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.ocr-resume-',dir=destination.parent) as temporary:
        root=Path(temporary).resolve()
        with zipfile.ZipFile(archive) as z:
            seen=set()
            for info in z.infolist():
                target=(root/info.filename).resolve()
                if (not target.is_relative_to(root) or target in seen
                        or '\\' in info.filename or stat.S_ISLNK(info.external_attr>>16)):
                    raise ValueError('Unsafe or duplicate resume archive path: '+info.filename)
                seen.add(target)
            z.extractall(root)
        outputs=[p.parent.parent for p in root.rglob('prepared/identity.json')]
        if len(outputs)!=1:raise ValueError('Archive must contain exactly one benchmark output root')
        return restore_outputs(outputs[0],destination)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True);parser.add_argument('--archive',required=True)
    args=parser.parse_args()
    print(export_bundle(args.output,args.archive))
