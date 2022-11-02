---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.0
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

### regex jeremy for daniel

```python
from fastcore.utils import *
import gradio as gr

def fix_imgs(src, dst):
    found = {n:(s,f) for n,s,f in
             re.findall(r'!\[(\S+)\|(\S+)\]\((\S+)\)', src)}

    def repl_img(x):
        res = found.get(x.group(1))
        if res:
            sz,nm = res
            return f'![{x.group(1)}|{sz}]({nm})'
        else: return f'MISSING IMAGE: {x.group(1)}'
    
#     result = re.sub(r'!\[\[([^.]+)\.\w+(\|\d+)?\]\]', repl_img, dst)
    result = re.sub(r'!\[\[(\S+)\.(png|jpeg)(\|\d+)?\]\]', repl_img, dst)
    return result

def fix_imgs_with_hide(src, dst):
    if dst != "":
        result_no_hide = fix_imgs(src, dst)
    else:
        result_no_hide = src
    
    def add_hide_top_func(x):
        add_hide_top = """\n\n[details='Images']\n\n"""
        return     f'{x.group(1)}{add_hide_top}{x.group(2)}'

    result_top_hide = re.sub(r'([\.|\?|\w|\`][\s]*)([\n]*!\[[^|]+\|\S+\]\(\S+\))', add_hide_top_func, result_no_hide)

    def add_hide_bottom_func(x):
        add_hide_bottom = """[/details]\n\n"""
        return     f'{x.group(1)}{add_hide_bottom}{x.group(2)}'

    return re.sub(r'(!\[[^|]+\|\S+\]\(\S+\)[\n]+)(\w|#)', add_hide_bottom_func, result_top_hide)


src_txt = Path('src.txt').read_text()
dst_txt = Path('dst.txt').read_text()

print(fix_imgs_with_hide(src_txt, dst_txt))



gr.Interface(fn=fix_imgs_with_hide, inputs=[gr.Textbox(lines=15),gr.Textbox(lines=15)], outputs=gr.Textbox(lines=15)).launch()
```

### regex \w, \d, [a-z], ?, +, *
`adfd-12184-*()`
`\d` :  has 5 matches  https://regex101.com/r/qae1H4/1
`\d+` : has 1 match https://regex101.com/r/OXMih8/1

`121adfd-*12d18dd4-*()_`
`\w*`: has 10 matches (so many because 0 or more == not `\w` or more `\w`) https://regex101.com/r/lk4VNo/1
`\w\w*`: has 3 matches  https://regex101.com/r/rfSbS9/1

`adfd-12184-*()_`
`\w`: has 10 matches https://regex101.com/r/KmRLIB/1
`\w+`: has 3 matches  https://regex101.com/r/VUBDnJ/1
`[a-z]` : has 4 matches https://regex101.com/r/klycje/1
`[a-z]+`: has 1 match https://regex101.com/r/FtFZeA/1

`121adfd-12184-*()_`
`[a-z]*`: match every character as 0 occurance is accepted too  https://regex101.com/r/0NbPzr/1

`121adfd-12d18dd4-*()_`
`d[a-z]*`: has 3 matches   https://regex101.com/r/gROAgR/1
`d[a-z]?`: has 4 matches   https://regex101.com/r/Ii4LYs/1

keep working https://www.youtube.com/watch?v=GyJtxd14DTc&loop=0
