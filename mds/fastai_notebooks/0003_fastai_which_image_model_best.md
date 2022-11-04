---
skip_exec: true
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

```python
from fastdebug.utils import *
from fastai.vision.all import *
```

```python
# fastnbs("DataBlock", "src", True)
```

# 0003_fastai_which_image_model_best


*The data, concept, and initial implementation of this notebook was done in Colab by Ross Wightman, the creator of timm. I (Jeremy Howard) did some refactoring, curating, and expanding of the analysis, and added prose.*


## timm

[PyTorch Image Models](https://timm.fast.ai/) (timm) is a wonderful library by Ross Wightman which provides state-of-the-art pre-trained computer vision models. It's like Huggingface Transformers, but for computer vision instead of NLP (and it's not restricted to transformers-based models)!

Ross has been kind enough to help me understand how to best take advantage of this library by identifying the top models. I'm going to share here so of what I've learned from him, plus some additional ideas.


## Access to pytorch-image-models repo


### ! git clone, %cd, %ls

Ross regularly benchmarks new models as they are added to timm, and puts the results in a CSV in the project's GitHub repo. To analyse the data, we'll first clone the repo:

```python
#| eval: false
# %cd ~/Documents
# ! git clone --depth 1 https://github.com/rwightman/pytorch-image-models.git
# ! git clone git@github.com:rwightman/pytorch-image-models.git --depth 1
```

```python
%cd ~/Documents/pytorch-image-models/results
%ls
```

### pd.read_csv('results-imagenet.csv')


Using Pandas, we can read the two CSV files we need, and merge them together.

```python
import pandas as pd
df_results = pd.read_csv('results-imagenet.csv')
```

```python
df_results.head()
```

```python
path = Path("/Users/Natsume/Documents/pytorch-image-models/results")
```

```python
for p in path.ls():
    if "infer" in p.name: 
        p
```

### df.iloc[2:5, [1, -2, -1] ] vs df.loc[df.model.str.contains("res"), ['model', 'family'] ]
how to merge data from benchmark-{part}-amp-nhwc-pt111-cu113-rtx3090.csv with data from results-imagenet.csv

how to turn a "infer_samples_per_sec" column into a speed "sec" column

how to use regex to extract model family from model names

how to use regex to select data of only specified different model families

```python
# @snoop
def get_data(part, col):
    "merge data from benchmark-{part}-amp-nhwc-pt111-cu113-rtx3090.csv with data from results-imagenet.csv"
#     pp(pd.read_csv(f'benchmark-{part}-amp-nhwc-pt111-cu113-rtx3090.csv').head())
#     pp(pd.read_csv(f'benchmark-{part}-amp-nhwc-pt111-cu113-rtx3090.csv').columns)
#     return
    df = pd.read_csv(f'benchmark-{part}-amp-nhwc-pt111-cu113-rtx3090.csv').merge(df_results, on='model')
#     pp(df.columns, df['model'])
#     return
    df['secs'] = 1. / df[col] # add a column named 'secs': the speed of making one inference
    df['family'] = df.model.str.extract('^([a-z]+?(?:v2)?)(?:\d|_|$)') # use regex to extract the model family name
    pp(df.model, df.model.str.extract('^([a-z]+?(?:v2)?)(?:\d|_|$)'))
#     pp(df["secs"], df["family"])
#     return
    df = df[~df.model.str.endswith('gn')] # only get models whose names don't end with 'gn'
#     pp(df.model)
#     return
#     pp(df.loc[df.model.str.contains('in22'),'family'])
#     pp(df.loc[df.model.str.contains('in22'),'family'] + '_in22')    
#     pp(df.loc[df.model.str.contains('in22'), ('model', 'family')])
#     return
    df.loc[df.model.str.contains('in22'),'family'] = df.loc[df.model.str.contains('in22'),'family'] + '_in22'
#     pp(df.loc[df.model.str.contains('in22'),'family'])
#     pp(df.loc[:, ['model', 'family']]) # access groups by named columns and rows
    df.loc[df.model.str.contains('resnet.*d'),'family'] = df.loc[df.model.str.contains('resnet.*d'),'family'] + 'd'
#     pp(df.iloc[:, [0, -2, -1]]) # iloc: group columns and rows using numbers not named strings
#     pp(df.loc[df.model.str.contains('resnet.*d'),'family'])
#     pp(df.loc[df.family.str.contains('^re[sg]netd?|beit|convnext|levit|efficient|vit|vgg'), "family"])
    return df[df.family.str.contains('^re[sg]netd?|beit|convnext|levit|efficient|vit|vgg')]
```

We'll also add a "family" column that will allow us to group architectures into categories with similar characteristics:

Ross has told me which models he's found the most usable in practice, so I'll limit the charts to just look at these. (I also include VGG, not because it's good, but as a comparison to show how far things have come in the last few years.)

```python
df = get_data('infer', 'infer_samples_per_sec')
```

## Regex
[cheatsheet](https://www.rexegg.com/regex-quickstart.html) by examples


### learn_regex

```python
import re
```

```python
import pandas as pd
```

```python
# @snoop
def learn_regex(egs, targets, *patterns, 
                comp=[1,2], # tuple, 1: the first pattern, 2: the second pattern and etc
                db=True):
    import pandas as pd
    lst = []
    for eg, fml in zip(egs, fmls):
        lst.append([eg, fml] + [re.findall(p, eg)[0] for p in patterns])
    df = pd.DataFrame(lst, columns=['egs', 'targets'] + [p for p in patterns])
    if db:
        comp = [c + 1 for c in comp]
        pp(comp)
        df_diff = df.loc[df.iloc[:,-2] != df.iloc[:,-1]]
        df_same = df.loc[df.iloc[:,-2] == df.iloc[:,-1]]
        assert len(egs) == len(df_diff) + len(df_same), "len(egs) == len(df_diff) + len(df_same) should be true, but false"
        return df_same, df_diff
    return df
```

### regex [^abc]?+*

```python
patterns = """
([^abc]?)
([^abc]+)
([^abc]*)
""".split("\n")[1:-1]
patterns

eg = "Anything but abc."

pd.DataFrame([[eg] + [re.findall(p, eg) for p in patterns]], columns=['eg'] + [p for p in patterns])
```

### regex look ahead (?=)

```python
patterns = [
'(?=\d{10})\d{5}', # there should be 10 consecutive digits, if there are, then get the first 5 digits
'\d{5}(?=\d{10})', # want to get 5 consecutive digits and afterwards there should be 10 digits ahead
'\d+(?= dollars)', # there should be ' dollars' ahead, and get all consecutive digits before it
'(?=\d+ dollars)\d+', # there should be 1 or more consecutive digits ahead + ' dollars', search for the digits
]

egs = [
'afa0123456789',
'adsf888880123456789',
'1adf12d100 dollars',
'100adf12d200 dollars',
'100adf12d200 dollars'    
]

pd.DataFrame([[eg, re.findall(p, eg)] for eg, p in zip(egs, patterns)])
pd.DataFrame([[eg, re.findall(p, eg)] for eg, p in zip(egs, patterns)], index = [p for p in patterns]).transpose()
```

```python

```

### Regex syntax summary


-   Character: All characters, except those having special meaning in regex, matches themselves. E.g., the regex `x` matches substring `"x"`; regex `9` matches `"9"`; regex `=` matches `"="`; and regex `@` matches `"@"`.
-   Special Regex Characters: These characters have special meaning in regex (to be discussed below): `.`, `+`, `*`, `?`, `^`, `$`, `(`, `)`, `[`, `]`, `{`, `}`, `|`, `\`.
-   Escape Sequences (\char):
    -   To match a character having special meaning in regex, you need to use a escape sequence prefix with a backslash (`\`). E.g., `\.` matches `"."`; regex `\+` matches `"+"`; and regex `\(` matches `"("`.
    -   You also need to use regex `\\` to match `"\"` (back-slash).
    -   Regex recognizes common escape sequences such as `\n` for newline, `\t` for tab, `\r` for carriage-return, `\nnn` for a up to 3-digit octal number, `\xhh` for a two-digit hex code, `\uhhhh` for a 4-digit Unicode, `\uhhhhhhhh` for a 8-digit Unicode.
-   A Sequence of Characters (or String): Strings can be matched via combining a sequence of characters (called sub-expressions). E.g., the regex `Saturday` matches `"Saturday"`. The matching, by default, is case-sensitive, but can be set to case-insensitive via _modifier_.
-   OR Operator (|): E.g., the regex `four|4` accepts strings `"four"` or `"4"`.
-   Character class (or Bracket List):
    -   [...]: Accept ANY ONE of the character within the square bracket, e.g., `[aeiou]` matches `"a"`, `"e"`, `"i"`, `"o"` or `"u"`.
    -   [.-.] (Range Expression): Accept ANY ONE of the character in the _range_, e.g., `[0-9]` matches any digit; `[A-Za-z]` matches any uppercase or lowercase letters.
    -   [^...]: NOT ONE of the character, e.g., `[^0-9]` matches any non-digit.
    -   Only these four characters require escape sequence inside the bracket list: `^`, `-`, `]`, `\`.
-   Occurrence Indicators (or Repetition Operators):
    -   +: one or more (`1+`), e.g., `[0-9]+` matches one or more digits such as `'123'`, `'000'`.
    -   *: zero or more (`0+`), e.g., `[0-9]*` matches zero or more digits. It accepts all those in `[0-9]+` plus the empty string.
    -   ?: zero or one (optional), e.g., `[+-]?` matches an optional `"+"`, `"-"`, or an empty string.
    -   {m,n}: `m` to `n` (both inclusive)
    -   {m}: exactly `m` times
    -   {m,}: `m` or more (`m+`)
-   Metacharacters: matches a character
    -   . (dot): ANY ONE character except newline. Same as `[^\n]`
    -   \d, \D: ANY ONE digit/non-digit character. Digits are `[0-9]`
    -   \w, \W: ANY ONE word/non-word character. For ASCII, word characters are `[a-zA-Z0-9_]`
    -   \s, \S: ANY ONE space/non-space character. For ASCII, whitespace characters are `[ \n\r\t\f]`
-   Position Anchors: does not match character, but position such as start-of-line, end-of-line, start-of-word and end-of-word.
    -   ^, $: start-of-line and end-of-line respectively. E.g., `^[0-9]$` matches a numeric string.
    -   \b: boundary of word, i.e., start-of-word or end-of-word. E.g., `\bcat\b` matches the word `"cat"` in the input string.
    -   \B: Inverse of \b, i.e., non-start-of-word or non-end-of-word.
    -   \<, \>: start-of-word and end-of-word respectively, similar to `\b`. E.g., `\<cat\>` matches the word `"cat"` in the input string.
    -   \A, \Z: start-of-input and end-of-input respectively.
-   Parenthesized Back References:
    -   Use parentheses `( )` to create a back reference.
    -   Use `$1`, `$2`, ... (Java, Perl, JavaScript) or `\1`, `\2`, ... (Python) to retreive the back references in sequential order.
-   Laziness (Curb Greediness for Repetition Operators): `*?`, `+?`, `??`, `{m,n}?`, `{m,}?`


### regex ? + * ``` [0-9]+ or \d+```

1.  A regex (_regular expression_) consists of a sequence of _sub-expressions_. In this example, `[0-9]` and `+`.
2.  The `[...]`, known as _character class_ (or _bracket list_), encloses a list of characters. It matches any SINGLE character in the list. In this example, `[0-9]` matches any SINGLE character between 0 and 9 (i.e., a digit), where dash (`-`) denotes the _range_.
3.  The `+`, known as _occurrence indicator_ (or _repetition operator_), indicates one or more occurrences (`1+`) of the previous sub-expression. In this case, `[0-9]+` matches one or more digits.
4.  A regex may match a portion of the input (i.e., substring) or the entire input. In fact, it could match zero or more substrings of the input (with global modifier).
5.  This regex matches any numeric substring (of digits 0 to 9) of the input. For examples,
    
    1.  If the input is `"abc123xyz"`, it matches substring `"123"`.
    2.  If the input is `"abcxyz"`, it matches nothing.
    3.  If the input is `"abc00123xyz456_0"`, it matches substrings `"00123"`, `"456"` and `"0"` (three matches).
    
    Take note that this regex matches number with leading zeros, such as `"000"`, `"0123"` and `"0001"`, which may not be desirable.
6.  You can also write `\d+`, where `\d` is known as a _metacharacter_ that matches any digit (same as `[0-9]`). There are more than one ways to write a regex! Take note that many programming languages (C, Java, JavaScript, Python) use backslash `\` as the prefix for escape sequences (e.g., `\n` for newline), and you need to write `"\\d+"` instead.


https://www3.ntu.edu.sg/home/ehchua/programming/howto/Regexe.html#zz-1.2

```python
import re   # Need module 're' for regular expression
# Try find: re.findall(regexStr, inStr) -> matchedSubstringsList
# r'...' denotes raw strings which ignore escape code, i.e., r'\n' is '\'+'n'
```

```python
re.findall(r'[0-9]', 'abc123xyz') # search from of 0-9, just 1
```

```python
re.findall(r'[0-9]?', 'abc123xyz') # search from 0-9, can be 0 or 1
```

```python
re.findall(r'[0-9]*', 'abc123xyz') # search from 0-9, can be 0 or many
```

```python
re.findall(r'[0-9]+', 'abc123xyz') # search from 0-9, can be 1 or many
```

```python
re.findall(r'[0-9]+', 'abcxyz')
```

```python
re.findall(r'[0-9]+', 'abc00123xyz456_0')
```

```python
re.findall(r'\d+', 'abc00123xyz456_0')
```

```python
# Try substitute: re.sub(regexStr, replacementStr, inStr) -> outStr
re.sub(r'[0-9]+', r'*', 'abc00123xyz456_0')
```

```python
# Try substitute with count: re.subn(regexStr, replacementStr, inStr) -> (outStr, count)
re.subn(r'[0-9]+', r'*', 'abc00123xyz456_0')
```

### regex ^ and $ ```^[0-9]+$ or ^\d+$```

1.  The leading `^` and the trailing `$` are known as _position anchors_, which match the start and end positions of the line, respectively. As the result, the entire input string shall be matched fully, instead of a portion of the input string (substring).

2.  This regex matches any non-empty numeric strings (comprising of digits 0 to 9), e.g., "`0`" and "`12345`". It does not match with "" (empty string), "`abc`", "`a123`", "`abc123xyz`", etc. However, it also matches "`000`", "`0123`" and "`0001`" with leading zeros.

```python
re.findall(r'^[0-9]+$', '837462729394')
```

```python
re.findall(r'^[0-9]+$', '83746a29394')
```

```python
re.findall(r'^\d+$', '3429873483743')
```

```python
re.findall(r'^\d+$', '3429873483b743')
```

```python

```

### regex | 0 [1-9] ``` [1-9][0-9]*|0 or [1-9]\d*|0```

1.  `[1-9]` matches any character between 1 to 9; `[0-9]*` matches zero or more digits. The `*` is an _occurrence indicator_ representing zero or more occurrences. Together, `[1-9][0-9]*` matches any numbers without a leading zero.
2.  `|` represents the OR operator; which is used to include the number `0`.
3.  This expression matches "`0`" and "`123`"; but does not match "`000`" and "`0123`" (but see below).
4.  You can replace `[0-9]` by metacharacter `\d`, but not `[1-9]`.
5.  We did not use _position anchors_ `^` and `$` in this regex. Hence, it can match any parts of the input string. For examples,
    1.  If the input string is "`abc123xyz`", it matches the substring `"123"`.
    2.  If the input string is `"abcxyz"`, it matches nothing.
    3.  If the input string is `"abc123xyz456_0"`, it matches substrings `"123"`, `"456"` and `"0"` (three matches).
    4.  If the input string is `"0012300"`, it matches substrings: `"0"`, `"0"` and `"12300"` (three matches)!!!

```python
re.findall(r'[1-9][0-9]*|0', 'afa93842782dfs01093adfd')
```

```python
re.findall(r'[1-9]\d*|0', 'afa93842782dfs01093adfd')
```

```python
re.findall(r'[1-9]\d*|0', '012143adf0343')
```

### regex ```^[+-]?[1-9][0-9]*|0$ or ^[+-]?[1-9]\d*|0$```

1.  This regex match an Integer literal (for entire string with the _position anchors_), both positive, negative and zero.
2.  `[+-]` matches either `+` or `-` sign. `?` is an _occurrence indicator_ denoting 0 or 1 occurrence, i.e. optional. Hence, `[+-]?` matches an optional leading `+` or `-` sign.
3.  We have covered three occurrence indicators: `+` for one or more, `*` for zero or more, and `?` for zero or one.

```python
re.findall(r'^[+-]?[1-9][0-9]*|0$', '+34adkfjs9284')
```

```python
re.findall(r'^[+-]?[1-9][0-9]*|0$', '-343410adkfjs92840')
```

```python
re.findall(r'^[+-]?[1-9][0-9]*|0$', '343410adkfjs92840')
```

```python

```

### ```[a-zA-Z_][0-9a-zA-Z_]* or [a-zA-Z_]\w*```

1.  Begin with one letters or underscore, followed by zero or more digits, letters and underscore.
2.  You can use _metacharacter_ `\w` for a word character `[a-zA-Z0-9_]`. Recall that _metacharacter_ `\d` can be used for a digit `[0-9]`.

```python
re.findall(r'[a-zA-Z_][0-9a-zA-Z_]*', '343410adkfjs92840')
```

```python
re.findall(r'[a-zA-Z_]\w*', '343410adkfjs92840')
```

```python

```

### regex on image filename ```^\w+\.(gif|png|jpg|jpeg)$```

1.  The _position anchors_ `^` and `$` match the beginning and the ending of the input string, respectively. That is, this regex shall match the entire input string, instead of a part of the input string (substring).
2.  `\w+` matches one or more word characters (same as `[a-zA-Z0-9_]+`).
3.  `\.` matches the dot `(.)` character. We need to use `\.` to represent `.` as `.` has special meaning in regex. The `\` is known as the escape code, which restore the original literal meaning of the following character. Similarly, `*`, `+`, `?` (occurrence indicators), `^`, `$` (position anchors) have special meaning in regex. You need to use an escape code to match with these characters.
4.  `(gif|png|jpg|jpeg)` matches either "`gif`", "`png`", "`jpg`" or "`jpeg`". The `|` denotes "OR" operator. The parentheses are used for grouping the selections.
5.  The _modifier_ `i` after the regex specifies case-insensitive matching (applicable to some languages like Perl and JavaScript only). That is, it accepts "`test.GIF`" and "`TesT.Gif`".

```python
re.findall(r'(^\w+)\.(gif|png|jpg|jpeg)$', 'thisisme.jpg')
```

```python
re.findall(r'(^\w*)\.(gif|png|jpg|jpeg)$', 'tHisisMe.jpg')
```

```python
re.findall(r'(^\w?)\.(gif|png|jpg|jpeg)$', 'thisisme.jpg')
```

```python

```

#### regex on email ```^\w+([.-]?\w+)*@\w+([.-]?\w+)*(\.\w{2,3})+$```

1.  The _position anchors_ `^` and `$` match the beginning and the ending of the input string, respectively. That is, this regex shall match the entire input string, instead of a part of the input string (substring).
2.  `\w+` matches 1 or more word characters (same as `[a-zA-Z0-9_]+`).
3.  `[.-]?` matches an optional character `.` or `-`. Although dot (`.`) has special meaning in regex, in a character class (square brackets) any characters except `^`, `-`, `]` or `\` is a literal, and do not require escape sequence.
4.  `([.-]?\w+)*` matches 0 or more occurrences of `[.-]?\w+`.
5.  The sub-expression `\w+([.-]?\w+)*` is used to match the username in the email, before the `@` sign. It begins with at least one word character `[a-zA-Z0-9_]`, followed by more word characters or `.` or `-`. However, a `.` or `-` must follow by a word character `[a-zA-Z0-9_]`. That is, the input string cannot begin with `.` or `-`; and cannot contain "`..`", "`--`", "`.-`" or "`-.`". Example of valid string are "`a.1-2-3`".
6.  The `@` matches itself. In regex, all characters other than those having special meanings matches itself, e.g., `a` matches `a`, `b` matches `b`, and etc.
7.  Again, the sub-expression `\w+([.-]?\w+)*` is used to match the email domain name, with the same pattern as the username described above.
8.  The sub-expression `\.\w{2,3}` matches a `.` followed by two or three word characters, e.g., "`.com`", "`.edu`", "`.us`", "`.uk`", "`.co`".
9.  `(\.\w{2,3})+` specifies that the above sub-expression could occur one or more times, e.g., "`.com`", "`.co.uk`", "`.edu.sg`" etc.

Exercise: Interpret this regex, which provide another representation of email address: `^[\w\-\.\+]+\@[a-zA-Z0-9\.\-]+\.[a-zA-z0-9]{2,4}$`.

```python
re.findall(r'^\w+([.-]?\w+)*@\w+([.-]?\w+)*(\.\w{2,3})+$', 'pystrategyexplorer@gmail.com')
```

```python
re.findall(r'(^\w+[.-]?\w+)*@(\w+[.-]?\w+)*\.(\w{2,3})+$', 'pystrategy-explorer@gov-ac.uk')
```

```python
re.findall(r'(^\w+[.-]?\w+)*\@(\w+[.-]?\w+)*\.(\w{2,3})+$', 'pystrategy.explorer@gov-ac.uk')
```

```python
re.findall(r'(^\w+[.-]?\w+[-.]?\w+)+\@(\w+[.-]?\w+)*\.(\w{2,3})+$', 'pystra.tegyexp.lorer@gov-ac.uk')
```

```python

```

#### Swapping Words using Parenthesized Back-References ```^(\S+)\s+(\S+)$ and $2 $1```

1.  The `^` and `$` match the beginning and ending of the input string, respectively.
2.  The `\s` (lowercase `s`) matches a whitespace (blank, tab `\t`, and newline `\r` or `\n`). On the other hand, the `\S+` (uppercase `S`) matches anything that is NOT matched by `\s`, i.e., non-whitespace. In regex, the uppercase metacharacter denotes the _inverse_ of the lowercase counterpart, for example, `\w` for word character and `\W` for non-word character; `\d` for digit and `\D` or non-digit.
3.  The above regex matches two words (without white spaces) separated by one or more whitespaces.
4.  Parentheses `()` have two meanings in regex:
    1.  to group sub-expressions, e.g., `(abc)*`
    2.  to provide a so-called _back-reference_ for capturing and extracting matches.
5.  The parentheses in `(\S+)`, called _parenthesized back-reference_, is used to extract the matched substring from the input string. In this regex, there are two `(\S+)`, match the first two words, separated by one or more whitespaces `\s+`. The two matched words are extracted from the input string and typically kept in special variables `$1` and `$2` (or `\1` and `\2` in Python), respectively.
6.  To swap the two words, you can access the special variables, and print "`$2 $1`" (via a programming language); or substitute operator "`s/(\S+)\s+(\S+)/$2 $1/`" (in Perl).

```python
re.findall(r'\w+', 'this is me')
```

```python
re.findall(r'\w+', 'this_is_me')
```

```python
re.findall(r'\w+', 'this9_IS_me0')
```

```python
re.findall(r'\w+\s?', 'this is me')
```

```python
re.findall(r'(\w+\s?)?', 'this is me')
```

```python
re.findall(r'^(\S+)\s+(\S+)$', 'this is me')
```

```python
re.findall(r'^\S+\s+\S+$', 'this isme')
```

```python
re.findall(r'^(\S+)\s+\S+$', 'this isme')
```

```python
re.findall(r'^\S+\s+(\S+)$', 'this isme')
```

```python
re.findall(r'^(\S+)\s+(\S+)$', 'this isme')
```

```python
re.findall(r'^(\S+)\s+(\S+)$', 'this isme')[0][0]
```

```python
re.findall(r'^(\S+)\s+(\S+)$', 'apple orange')
```

```python
re.sub(r'^(\S+)\s+(\S+)$', r'\2 \1', 'apple orange')   # Prefix r for raw string which ignores escape
```

```python
re.sub(r'^(\S+)\s+(\S+)$', '\\2 \\1', 'apple orange')  # Need to use \\ for \ for regular string
```

```python

```

### regex on HTTP Addresses ```^http:\/\/\S+(\/\S+)*(\/)?$```

1.  Begin with `http://`. Take note that you may need to write `/` as `\/` with an escape code in some languages (JavaScript, Perl).
2.  Followed by `\S+`, one or more non-whitespaces, for the domain name.
3.  Followed by `(\/\S+)*`, zero or more "/...", for the sub-directories.
4.  Followed by `(\/)?`, an optional (0 or 1) trailing `/`, for directory request.

```python
re.findall(r'^http:\/\/\S+(\/\S+)*(\/)?$', 'https://github.com/firasdib/Regex101/wiki')
```

```python
re.findall(r'^https:\/\/(\w+\.\w+)\/$', 'https://github.com/')
```

```python
re.findall(r'^https:\/\/(\S+)$', 'https://github.com/')
```

```python
re.findall(r'^https:\/\/(\w+\.\w+)\/(\w+\/)+$', 'https://github.com/firasdib/Regex101/')
```

### 2.  Regular Expression (Regex) Syntax

A Regular Expression (or Regex) is a _pattern_ (or _filter_) that describes a set of strings that matches the pattern. In other words, a regex _accepts_ a certain set of strings and _rejects_ the rest.

A regex consists of a sequence of characters, metacharacters (such as `.`, `\d`, `\D`, `\`s, `\S`, `\w`, `\W`) and operators (such as `+`, `*`, `?`, `|`, `^`). They are constructed by combining many smaller sub-expressions.

#### 2.1  Matching a Single Character

The fundamental building blocks of a regex are patterns that match a _single_ character. Most characters, including all letters (`a-z` and `A-Z`) and digits (`0-9`), match itself. For example, the regex `x` matches substring `"x"`; `z` matches `"z"`; and `9` matches `"9"`.

Non-alphanumeric characters without special meaning in regex also matches itself. For example, `=` matches `"="`; `@` matches `"@"`.

#### 2.2  Regex Special Characters and Escape Sequences

##### Regex's Special Characters

These characters have special meaning in regex (I will discuss in detail in the later sections):

-   metacharacter: dot (`.`)
-   bracket list: `[ ]`
-   position anchors: `^`, `$`
-   occurrence indicators: `+`, `*`, `?`, `{ }`
-   parentheses: `( )`
-   or: `|`
-   escape and metacharacter: backslash (`\`)

##### Escape Sequences

The characters listed above have special meanings in regex. To match these characters, we need to prepend it with a backslash (`\`), known as _escape sequence_.  For examples, `\+` matches `"+"`; `\[` matches `"["`; and `\.` matches `"."`.

Regex also recognizes common escape sequences such as `\n` for newline, `\t` for tab, `\r` for carriage-return, `\nnn` for a up to 3-digit octal number, `\xhh` for a two-digit hex code, `\uhhhh` for a 4-digit Unicode, `\uhhhhhhhh` for a 8-digit Unicode.


https://www3.ntu.edu.sg/home/ehchua/programming/howto/Regexe.html#zz-1.3

```python

```

## Compare models' inference ability


### plotly.express.scatter(df, width, height, title, size, x, y, log_x, color, hover_name, hover_data)


Here's the results for inference performance (see the last section for training performance). In this chart:

- the x axis shows how many seconds it takes to process one image (**note**: it's a log scale)
- the y axis is the accuracy on Imagenet
- the size of each bubble is proportional to the size of images used in testing
- the color shows what "family" the architecture is from.

Hover your mouse over a marker to see details about the model. Double-click in the legend to display just one family. Single-click in the legend to show or hide a family.

**Note**: on my screen, Kaggle cuts off the family selector and some plotly functionality -- to see the whole thing, collapse the table of contents on the right by clicking the little arrow to the right of "*Contents*".

```python
# !pip install plotly
```

```python
import plotly.express as px
w,h = 1000,800

def show_all(df, title, size):
    pp(list(df.columns))
    return px.scatter(df, width=w, height=h, title=title,
        size=df[size]**2, # the size of bubble is dependent on the size of image for inference
        x='secs',  # how fast is the model for inference, left fast, right slow
        y='top1',  # how good/accurate is the model at inference, top best, bottom worse
        log_x=True, # make x log to smooth the difference
        color='family', # make the same family the same color
        hover_name='model', # when hover, the name of model is shown
        hover_data=[size]) # add df[size] column as another data to be displayed when hovering
```

```python
show_all(df, 'Inference', 'infer_img_size')
```

### plotly.express.scatter with trendline on selected model families


That number of families can be a bit overwhelming, so I'll just pick a subset which represents a single key model from each of the families that are looking best in our plot. I've also separated convnext models into those which have been pretrained on the larger 22,000 category imagenet sample (`convnext_in22`) vs those that haven't (`convnext`). (Note that many of the best performing models were trained on the larger sample -- see the papers for details before coming to conclusions about the effectiveness of these architectures more generally.)

```python
#     return df[df.family.str.contains('^re[sg]netd?|beit|convnext|levit|efficient|vit|vgg')]
```

```python
subs = 'levit|resnetd?|regnetx|vgg|convnext.*|efficientnetv2|beit'
```

In this chart, I'll add lines through the points of each family, to help see how they compare -- but note that we can see that a linear fit isn't actually ideal here! It's just there to help visually see the groups.

```python
def show_subs(df, title, size):
    df_subs = df[df.family.str.fullmatch(subs)]
    pp(df_subs.family.unique())
    return px.scatter(df_subs, width=w, height=h,  title=title,
        size=df_subs[size]**2,
        trendline="ols", # see docs for details on the line options
        trendline_options={'log_x':True}, # how to do trendline on scatter plots
        x='secs',  y='top1', log_x=True, color='family', hover_name='model', hover_data=[size])
```

```python
# help(px.scatter)
```

```python
# !pip install statsmodels
```

```python
show_subs(df, 'Inference', 'infer_img_size')
```

From this, we can see that the *levit* family models are extremely fast for image recognition, and clearly the most accurate amongst the faster models. 

That's not surprising, since these models are a hybrid of the best ideas from CNNs and transformers, so get the benefit of each. 

In fact, we see a similar thing even in the middle category of speeds -- the best is the ConvNeXt, which is a pure CNN, but which takes advantage of ideas from the transformers literature.

For the slowest models, *beit* is the most accurate -- although we need to be a bit careful of interpreting this, since it's trained on a larger dataset (ImageNet-21k, which is also used for *vit* models).


### plotly speed vs parameters


I'll add one other plot here, which is of speed vs parameter count. 

Often, parameter count is used in papers as a proxy for speed. 

However, as we see, there is a wide variation in speeds at each level of parameter count, so it's really not a useful proxy.

(Parameter count may be be useful for identifying how much memory a model needs, but even for that it's not always a great proxy.)

```python
px.scatter(df, width=w, height=h,
    x='param_count_x',  y='secs', 
    log_x=True, log_y=True, 
    color= 'infer_img_size',
    hover_name='model', hover_data=['infer_samples_per_sec', 'family']
)
```

```python
df_subs = df[df.family.str.fullmatch(subs)]
```

```python
px.scatter(df_subs, width=w, height=h,
    x='param_count_x',  y='secs', 
    log_x=True, 
    log_y=True, 
    trendline="ols", # see docs for details on the line options
    trendline_options={'log_x':True, 'log_y':True}, # how to do trendline on scatter plots           
    color= 'family', # 'infer_img_size',
    hover_name='model', hover_data=['infer_samples_per_sec', 'family']
)
```

## Training results


We'll now replicate the above analysis for training performance. First we grab the data:

```python
tdf = get_data('train', 'train_samples_per_sec')
```

Now we can repeat the same *family* plot we did above:

```python
show_all(tdf, 'Training', 'train_img_size')
```

...and we'll also look at our chosen subset of models:

```python
show_subs(tdf, 'Training', 'train_img_size')
```

Finally, we should remember that speed depends on hardware. If you're using something other than a modern NVIDIA GPU, your results may be different. In particular, I suspect that transformers-based models might have worse performance in general on CPUs (although I need to study this more to be sure).


### convert ipynb to md

```python
from fastdebug.utils import *
```

```python
# fastnbs("export model")
```

```python
doc(fastlistnbs)
```

```python
show_doc
```

```python

```
