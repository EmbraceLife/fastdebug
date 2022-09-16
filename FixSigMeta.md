{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "8a732f12-706d-4084-858b-25ab75931b76",
   "metadata": {},
   "source": [
    "# FixSigMeta"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "50498ed4",
   "metadata": {},
   "source": [
    "The goal of this notebook is to explore how `FixSigMeta` avoid problems when `Foo` wants to get signature from its `__init__` using `inspect.signature`. \n",
    "\n",
    "In order to understand the problems `FixSigMeta` is fixing, please read official [docs](https://fastcore.fast.ai/meta.html#fixsigmeta) first\n",
    "\n",
    "At the beginning, I know very little of the source code of `inspect.signature` or in fact `inspect._signature_from_callable` and have no idea how `FixSigMeta` enable `Foo` to overcome the potential problems.\n",
    "\n",
    "`Fastdebug` library and its `Fastdb.dbprint` enables me to debug any source code and evaluate the expressions you write sitting above the source code and `Fastdb.print` can display source code with comments I add when debugging with `dbprint`.\n",
    "\n",
    "At the end of the notebook, I hope to have a nice and detailed document on the exploration and have a in-depth understanding of how both `_signature_from_callable` and `FixSigMeta` work.\n",
    "\n",
    "Here is what I learnt from this notebook about how `FixSigMeta` solve the potential problems \n",
    "\n",
    "\n",
    "> As a metaclass, `FixSigMeta` defines its `__new__` to creates a class instance `Foo` with attribute `__signature__` and store the signature of `__init__`. This way `inspect._signature_from_callable` can directly help `Foo` to get signature from `__signature__` instead of going into `__new__`, `__call__` looking for signatures where the potential problems reside."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5c5c993a",
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "markdown",
   "id": "12d53179",
   "metadata": {},
   "source": [
    "## Having a better view in Jupyter notebook"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "9029dccb-9cad-45a9-b4e1-0b445bf7eb3d",
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# from IPython.core.display import display, HTML # a depreciated import\n",
    "from IPython.display import display, HTML "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "9b940390-ce64-471d-9ccc-811cb1589912",
   "metadata": {
    "collapsed": true
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<style>.container { width:100% !important; }</style>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "display(HTML(\"<style>.container { width:100% !important; }</style>\"))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "7b4846c0",
   "metadata": {},
   "source": [
    "## Explore `inspect.signature`"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "9a1f1988",
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "import inspect \n",
    "from inspect import *"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "52a5bd62",
   "metadata": {},
   "source": [
    "### `Foo` borrows signature from `Foo.__init__`"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "eb506571",
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "class Foo:\n",
    "    def __init__(self, a, b, c): pass\n",
    "\n",
    "    @classmethod\n",
    "    def clsmed(): pass"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "eb25afc6",
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "markdown",
   "id": "f74714f5",
   "metadata": {},
   "source": [
    "## Documenting `inspect.signature`"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "ca6f759f",
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "from fastdebug.core import *"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "657f04eb",
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "fdb = Fastdb(signature)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "1c2d6631",
   "metadata": {
    "collapsed": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "def signature(obj, *, follow_wrapped=True):===============================================(0)       \n",
      "    \"\"\"Get a signature object for the passed callable.\"\"\"=================================(1)       \n",
      "    return Signature.from_callable(obj, follow_wrapped=follow_wrapped)====================(2)       \n",
      "                                                                                                                                                        (3)\n"
     ]
    }
   ],
   "source": [
    "fdb.print()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "2a230110",
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "fdb.takExample(\"signature(Foo)\", signature=signature, Foo=Foo)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "78a76900",
   "metadata": {
    "collapsed": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "=========================================================     Investigating \u001b[91msignature\u001b[0m     ==========================================================\n",
      "================================================================     on line \u001b[91m2\u001b[0m     =================================================================\n",
      "=======================================================     with example \u001b[91msignature(Foo)\u001b[0m     ========================================================\n",
      "\n",
      "def signature(obj, *, follow_wrapped=True):                                                                                                             (0)\n",
      "    \"\"\"Get a signature object for the passed callable.\"\"\"                                                                                               (1)\n",
      "    return Signature.from_callable(obj, follow_wrapped=follow_wrapped)==================================================================================(2)\n",
      "                                                                                                                                            \u001b[91mLet's dive deeper\u001b[0m\n",
      "                                                                                                                                                        (3)\n",
      "\n",
      "====================================================================================================================\u001b[91mStart of my srcline exploration:\u001b[0m\n",
      "\n",
      "\n",
      "                                                                                                         Signature => Signature : <class 'inspect.Signature'>\n",
      "\n",
      "\n",
      "                                   Signature.from_callable => Signature.from_callable : <bound method Signature.from_callable of <class 'inspect.Signature'>>\n",
      "\n",
      "\n",
      "                                                                                                                      follow_wrapped => follow_wrapped : True\n",
      "======================================================================================================================\u001b[91mEnd of my srcline exploration:\u001b[0m\n",
      "\n",
      "<Signature (a, b, c)>\n",
      "\n",
      "\u001b[93mReview srcode with all comments added so far\u001b[0m========================================================================================================\n",
      "def signature(obj, *, follow_wrapped=True):===============================================(0)       \n",
      "    \"\"\"Get a signature object for the passed callable.\"\"\"=================================(1)       \n",
      "    return Signature.from_callable(obj, follow_wrapped=follow_wrapped)====================(2) # Let's dive deeper\n",
      "                                                                                                                                                        (3)\n",
      "                                                                                                                                     part No.1 out of 1 parts\n",
      "\n"
     ]
    }
   ],
   "source": [
    "fdb.dbprint(2, \"Let's dive deeper\", \"Signature\", \"Signature.from_callable\", \"follow_wrapped\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "227b75d9",
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "from pprint import pprint"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "c077135c",
   "metadata": {
    "collapsed": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "('    @classmethod\\n'\n",
      " '    def from_callable(cls, obj, *, follow_wrapped=True):\\n'\n",
      " '        \"\"\"Constructs Signature for the given callable object.\"\"\"\\n'\n",
      " '        return _signature_from_callable(obj, sigcls=cls,\\n'\n",
      " '                                        '\n",
      " 'follow_wrapper_chains=follow_wrapped)\\n')\n"
     ]
    }
   ],
   "source": [
    "pprint(inspect.getsource(Signature.from_callable))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "a672376d",
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "from inspect import _signature_from_callable"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "22a0f43c",
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "fdb = Fastdb(_signature_from_callable)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "7e4ca598",
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "fdb.takExample(\"_signature_from_callable(Foo, sigcls=cls, follow_wrapper_chains=follow_wrapped)\", Foo=Foo, cls=Signature, follow_wrapped=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "fc1cb54b",
   "metadata": {
    "collapsed": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "def _signature_from_callable(obj, *,======================================================(0)       \n",
      "                             follow_wrapper_chains=True,==================================(1)       \n",
      "                             skip_bound_arg=True,=========================================(2)       \n",
      "                             sigcls):=====================================================(3)       \n",
      "                                                                                                                                                        (4)\n",
      "    \"\"\"Private helper function to get signature for arbitrary=============================(5)       \n",
      "    callable objects.=====================================================================(6)       \n",
      "    \"\"\"===================================================================================(7)       \n",
      "                                                                                                                                                        (8)\n",
      "    _get_signature_of = functools.partial(_signature_from_callable,=======================(9)       \n",
      "                                follow_wrapper_chains=follow_wrapper_chains,==============(10)      \n",
      "                                skip_bound_arg=skip_bound_arg,============================(11)      \n",
      "                                sigcls=sigcls)============================================(12)      \n",
      "                                                                                                                                                        (13)\n",
      "    if not callable(obj):=================================================================(14)      \n",
      "        raise TypeError('{!r} is not a callable object'.format(obj))======================(15)      \n",
      "                                                                                                                                                        (16)\n",
      "    if isinstance(obj, types.MethodType):=================================================(17)      \n",
      "        # In this case we skip the first parameter of the underlying======================(18)      \n",
      "        # function (usually `self` or `cls`).=============================================(19)      \n",
      "        sig = _get_signature_of(obj.__func__)=============================================(20)      \n",
      "                                                                                                                                                        (21)\n",
      "        if skip_bound_arg:================================================================(22)      \n",
      "            return _signature_bound_method(sig)===========================================(23)      \n",
      "        else:=============================================================================(24)      \n",
      "            return sig====================================================================(25)      \n",
      "                                                                                                                                                        (26)\n",
      "    # Was this function wrapped by a decorator?===========================================(27)      \n",
      "    if follow_wrapper_chains:=============================================================(28)      \n",
      "        obj = unwrap(obj, stop=(lambda f: hasattr(f, \"__signature__\")))===================(29)      \n",
      "        if isinstance(obj, types.MethodType):=============================================(30)      \n",
      "            # If the unwrapped object is a *method*, we might want to=====================(31)      \n",
      "            # skip its first parameter (self).============================================(32)      \n",
      "            # See test_signature_wrapped_bound_method for details.========================(33)      \n",
      "            return _get_signature_of(obj)=================================================(34)      \n",
      "                                                                                                                                                        (35)\n",
      "    try:==================================================================================(36)      \n",
      "        sig = obj.__signature__===========================================================(37)      \n",
      "    except AttributeError:================================================================(38)      \n",
      "        pass==============================================================================(39)      \n",
      "    else:=================================================================================(40)      \n",
      "        if sig is not None:===============================================================(41)      \n",
      "            if not isinstance(sig, Signature):============================================(42)      \n",
      "                raise TypeError(==========================================================(43)      \n",
      "                    'unexpected object {!r} in __signature__ '============================(44)      \n",
      "                    'attribute'.format(sig))==============================================(45)      \n",
      "            return sig====================================================================(46)      \n",
      "                                                                                                                                                        (47)\n",
      "    try:==================================================================================(48)      \n",
      "        partialmethod = obj._partialmethod================================================(49)      \n",
      "    except AttributeError:================================================================(50)      \n",
      "        pass==============================================================================(51)      \n",
      "    else:=================================================================================(52)      \n",
      "        if isinstance(partialmethod, functools.partialmethod):============================(53)      \n",
      "            # Unbound partialmethod (see functools.partialmethod)=========================(54)      \n",
      "            # This means, that we need to calculate the signature=========================(55)      \n",
      "            # as if it's a regular partial object, but taking into========================(56)      \n",
      "            # account that the first positional argument==================================(57)      \n",
      "            # (usually `self`, or `cls`) will not be passed===============================(58)      \n",
      "            # automatically (as for boundmethods)=========================================(59)      \n",
      "                                                                                                                                                        (60)\n",
      "            wrapped_sig = _get_signature_of(partialmethod.func)===========================(61)      \n",
      "                                                                                                                                                        (62)\n",
      "            sig = _signature_get_partial(wrapped_sig, partialmethod, (None,))=============(63)      \n",
      "            first_wrapped_param = tuple(wrapped_sig.parameters.values())[0]===============(64)      \n",
      "            if first_wrapped_param.kind is Parameter.VAR_POSITIONAL:======================(65)      \n",
      "                # First argument of the wrapped callable is `*args`, as in================(66)      \n",
      "                # `partialmethod(lambda *args)`.==========================================(67)      \n",
      "                return sig================================================================(68)      \n",
      "            else:=========================================================================(69)      \n",
      "                sig_params = tuple(sig.parameters.values())===============================(70)      \n",
      "                assert (not sig_params or=================================================(71)      \n",
      "                        first_wrapped_param is not sig_params[0])=========================(72)      \n",
      "                new_params = (first_wrapped_param,) + sig_params==========================(73)      \n",
      "                return sig.replace(parameters=new_params)=================================(74)      \n",
      "                                                                                                                                                        (75)\n",
      "    if isfunction(obj) or _signature_is_functionlike(obj):================================(76)      \n",
      "        # If it's a pure Python function, or an object that is duck type==================(77)      \n",
      "        # of a Python function (Cython functions, for instance), then:====================(78)      \n",
      "        return _signature_from_function(sigcls, obj,======================================(79)      \n",
      "                                        skip_bound_arg=skip_bound_arg)====================(80)      \n",
      "                                                                                                                                                        (81)\n",
      "    if _signature_is_builtin(obj):========================================================(82)      \n",
      "        return _signature_from_builtin(sigcls, obj,=======================================(83)      \n",
      "                                       skip_bound_arg=skip_bound_arg)=====================(84)      \n",
      "                                                                                                                                                        (85)\n",
      "    if isinstance(obj, functools.partial):================================================(86)      \n",
      "        wrapped_sig = _get_signature_of(obj.func)=========================================(87)      \n",
      "        return _signature_get_partial(wrapped_sig, obj)===================================(88)      \n",
      "                                                                                                                                                        (89)\n",
      "    sig = None============================================================================(90)      \n",
      "    if isinstance(obj, type):=============================================================(91)      \n",
      "        # obj is a class or a metaclass===================================================(92)      \n",
      "                                                                                                                                                        (93)\n",
      "        # First, let's see if it has an overloaded __call__ defined=======================(94)      \n",
      "        # in its metaclass================================================================(95)      \n",
      "        call = _signature_get_user_defined_method(type(obj), '__call__')==================(96)      \n",
      "        if call is not None:==============================================================(97)      \n",
      "            sig = _get_signature_of(call)=================================================(98)      \n",
      "        else:=============================================================================(99)      \n",
      "            factory_method = None=========================================================(100)     \n",
      "            new = _signature_get_user_defined_method(obj, '__new__')======================(101)     \n",
      "            init = _signature_get_user_defined_method(obj, '__init__')====================(102)     \n",
      "            # Now we check if the 'obj' class has an own '__new__' method=================(103)     \n",
      "            if '__new__' in obj.__dict__:=================================================(104)     \n",
      "                factory_method = new======================================================(105)     \n",
      "            # or an own '__init__' method=================================================(106)     \n",
      "            elif '__init__' in obj.__dict__:==============================================(107)     \n",
      "                factory_method = init=====================================================(108)     \n",
      "            # If not, we take inherited '__new__' or '__init__', if present===============(109)     \n",
      "            elif new is not None:=========================================================(110)     \n",
      "                factory_method = new======================================================(111)     \n",
      "            elif init is not None:========================================================(112)     \n",
      "                factory_method = init=====================================================(113)     \n",
      "                                                                                                                                                        (114)\n",
      "            if factory_method is not None:================================================(115)     \n",
      "                sig = _get_signature_of(factory_method)===================================(116)     \n",
      "                                                                                                                                                        (117)\n",
      "        if sig is None:===================================================================(118)     \n",
      "            # At this point we know, that `obj` is a class, with no user-=================(119)     \n",
      "            # defined '__init__', '__new__', or class-level '__call__'====================(120)     \n",
      "                                                                                                                                                        (121)\n",
      "            for base in obj.__mro__[:-1]:=================================================(122)     \n",
      "                # Since '__text_signature__' is implemented as a==========================(123)     \n",
      "                # descriptor that extracts text signature from the========================(124)     \n",
      "                # class docstring, if 'obj' is derived from a builtin=====================(125)     \n",
      "                # class, its own '__text_signature__' may be 'None'.======================(126)     \n",
      "                # Therefore, we go through the MRO (except the last=======================(127)     \n",
      "                # class in there, which is 'object') to find the first====================(128)     \n",
      "                # class with non-empty text signature.====================================(129)     \n",
      "                try:======================================================================(130)     \n",
      "                    text_sig = base.__text_signature__====================================(131)     \n",
      "                except AttributeError:====================================================(132)     \n",
      "                    pass==================================================================(133)     \n",
      "                else:=====================================================================(134)     \n",
      "                    if text_sig:==========================================================(135)     \n",
      "                        # If 'base' class has a __text_signature__ attribute:=============(136)     \n",
      "                        # return a signature based on it==================================(137)     \n",
      "                        return _signature_fromstr(sigcls, base, text_sig)=================(138)     \n",
      "                                                                                                                                                        (139)\n",
      "            # No '__text_signature__' was found for the 'obj' class.======================(140)     \n",
      "            # Last option is to check if its '__init__' is================================(141)     \n",
      "            # object.__init__ or type.__init__.===========================================(142)     \n",
      "            if type not in obj.__mro__:===================================================(143)     \n",
      "                # We have a class (not metaclass), but no user-defined====================(144)     \n",
      "                # __init__ or __new__ for it==============================================(145)     \n",
      "                if (obj.__init__ is object.__init__ and===================================(146)     \n",
      "                    obj.__new__ is object.__new__):=======================================(147)     \n",
      "                    # Return a signature of 'object' builtin.=============================(148)     \n",
      "                    return sigcls.from_callable(object)===================================(149)     \n",
      "                else:=====================================================================(150)     \n",
      "                    raise ValueError(=====================================================(151)     \n",
      "                        'no signature found for builtin type {!r}'.format(obj))===========(152)     \n",
      "                                                                                                                                                        (153)\n",
      "    elif not isinstance(obj, _NonUserDefinedCallables):===================================(154)     \n",
      "        # An object with __call__=========================================================(155)     \n",
      "        # We also check that the 'obj' is not an instance of==============================(156)     \n",
      "        # _WrapperDescriptor or _MethodWrapper to avoid===================================(157)     \n",
      "        # infinite recursion (and even potential segfault)================================(158)     \n",
      "        call = _signature_get_user_defined_method(type(obj), '__call__')==================(159)     \n",
      "        if call is not None:==============================================================(160)     \n",
      "            try:==========================================================================(161)     \n",
      "                sig = _get_signature_of(call)=============================================(162)     \n",
      "            except ValueError as ex:======================================================(163)     \n",
      "                msg = 'no signature found for {!r}'.format(obj)===========================(164)     \n",
      "                raise ValueError(msg) from ex=============================================(165)     \n",
      "                                                                                                                                                        (166)\n",
      "    if sig is not None:===================================================================(167)     \n",
      "        # For classes and objects we skip the first parameter of their====================(168)     \n",
      "        # __call__, __new__, or __init__ methods==========================================(169)     \n",
      "        if skip_bound_arg:================================================================(170)     \n",
      "            return _signature_bound_method(sig)===========================================(171)     \n",
      "        else:=============================================================================(172)     \n",
      "            return sig====================================================================(173)     \n",
      "                                                                                                                                                        (174)\n",
      "    if isinstance(obj, types.BuiltinFunctionType):========================================(175)     \n",
      "        # Raise a nicer error message for builtins========================================(176)     \n",
      "        msg = 'no signature found for builtin function {!r}'.format(obj)==================(177)     \n",
      "        raise ValueError(msg)=============================================================(178)     \n",
      "                                                                                                                                                        (179)\n",
      "    raise ValueError('callable {!r} is not supported by signature'.format(obj))===========(180)     \n",
      "                                                                                                                                                        (181)\n"
     ]
    }
   ],
   "source": [
    "fdb.print()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "c4d87905",
   "metadata": {
    "collapsed": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "==================================================     Investigating \u001b[91m_signature_from_callable\u001b[0m     ==================================================\n",
      "================================================================     on line \u001b[91m14\u001b[0m     ================================================================\n",
      "=======================     with example \u001b[91m_signature_from_callable(Foo, sigcls=cls, follow_wrapper_chains=follow_wrapped)\u001b[0m     =======================\n",
      "\n",
      "                                sigcls=sigcls)                                                                                                          (12)\n",
      "                                                                                                                                                        (13)\n",
      "    if not callable(obj):===============================================================================================================================(14)\n",
      "                                                                                                                                         \u001b[91mwhat is the locals()\u001b[0m\n",
      "        raise TypeError('{!r} is not a callable object'.format(obj))                                                                                    (15)\n",
      "                                                                                                                                                        (16)\n",
      "\n",
      "====================================================================================================================\u001b[91mStart of my srcline exploration:\u001b[0m\n",
      "\n",
      "\n",
      "_get_signature_of => _get_signature_of : functools.partial(<function _signature_from_callable at 0x1053a4940>, follow_wrapper_chains=True, skip_bound_arg=True, sigcls=<class 'inspect.Signature'>)\n",
      "\n",
      "\n",
      "{'c': 'pprint(locals())',\n",
      " 'codes': ('_get_signature_of', 'pprint(locals())'),\n",
      " 'env': {'_get_signature_of': functools.partial(<function _signature_from_callable at 0x1053a4940>, follow_wrapper_chains=True, skip_bound_arg=True, sigcls=<class 'inspect.Signature'>),\n",
      "         'follow_wrapper_chains': True,\n",
      "         'obj': <class '__main__.Foo'>,\n",
      "         'sigcls': <class 'inspect.Signature'>,\n",
      "         'skip_bound_arg': True},\n",
      " 'output': '_get_signature_of => _get_signature_of : '\n",
      "           'functools.partial(<function _signature_from_callable at '\n",
      "           '0x1053a4940>, follow_wrapper_chains=True, skip_bound_arg=True, '\n",
      "           \"sigcls=<class 'inspect.Signature'>)\"}\n",
      "                                                                                                                  pprint(locals()) => pprint(locals()) : None\n",
      "======================================================================================================================\u001b[91mEnd of my srcline exploration:\u001b[0m\n",
      "\n",
      "<Signature (a, b, c)>\n",
      "\n",
      "\u001b[93mReview srcode with all comments added so far\u001b[0m========================================================================================================\n",
      "def _signature_from_callable(obj, *,======================================================(0)       \n",
      "                             follow_wrapper_chains=True,==================================(1)       \n",
      "                             skip_bound_arg=True,=========================================(2)       \n",
      "                             sigcls):=====================================================(3)       \n",
      "                                                                                                                                                        (4)\n",
      "    \"\"\"Private helper function to get signature for arbitrary=============================(5)       \n",
      "    callable objects.=====================================================================(6)       \n",
      "    \"\"\"===================================================================================(7)       \n",
      "                                                                                                                                                        (8)\n",
      "    _get_signature_of = functools.partial(_signature_from_callable,=======================(9)       \n",
      "                                follow_wrapper_chains=follow_wrapper_chains,==============(10)      \n",
      "                                skip_bound_arg=skip_bound_arg,============================(11)      \n",
      "                                sigcls=sigcls)============================================(12)      \n",
      "                                                                                                                                                        (13)\n",
      "    if not callable(obj):=================================================================(14) # what is the locals()\n",
      "        raise TypeError('{!r} is not a callable object'.format(obj))======================(15)      \n",
      "                                                                                                                                                        (16)\n",
      "    if isinstance(obj, types.MethodType):=================================================(17)      \n",
      "        # In this case we skip the first parameter of the underlying======================(18)      \n",
      "        # function (usually `self` or `cls`).=============================================(19)      \n",
      "        sig = _get_signature_of(obj.__func__)=============================================(20)      \n",
      "                                                                                                                                                        (21)\n",
      "        if skip_bound_arg:================================================================(22)      \n",
      "            return _signature_bound_method(sig)===========================================(23)      \n",
      "        else:=============================================================================(24)      \n",
      "            return sig====================================================================(25)      \n",
      "                                                                                                                                                        (26)\n",
      "    # Was this function wrapped by a decorator?===========================================(27)      \n",
      "    if follow_wrapper_chains:=============================================================(28)      \n",
      "        obj = unwrap(obj, stop=(lambda f: hasattr(f, \"__signature__\")))===================(29)      \n",
      "        if isinstance(obj, types.MethodType):=============================================(30)      \n",
      "            # If the unwrapped object is a *method*, we might want to=====================(31)      \n",
      "            # skip its first parameter (self).============================================(32)      \n",
      "                                                                                                                                     part No.1 out of 6 parts\n",
      "\n"
     ]
    }
   ],
   "source": [
    "fdb.dbprint(14, \"what is the locals()\", \"_get_signature_of\", \"pprint(locals())\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6261ee57",
   "metadata": {},
   "source": [
    "### `Base.__new__` stops `Foo` borrows signature from `Foo.__init__`"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d852701d",
   "metadata": {
    "collapsed": true,
    "hide_input": false
   },
   "outputs": [],
   "source": [
    "class Base: # pass\n",
    "    def __new__(self, **args): pass  # defines a __new__ \n",
    "\n",
    "class Foo(Base):\n",
    "    def __init__(self, d, e, f): pass\n",
    "    \n",
    "inspect.signature(Foo) # not a problem for python 3.9+, but is a problem for python 3.7"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7362d7d5",
   "metadata": {
    "code_folding": [
     1
    ],
    "collapsed": true,
    "hide_input": false
   },
   "outputs": [],
   "source": [
    "sigOld=\"\"\"\n",
    "def _signature_from_callableOld(obj, *,\n",
    "                             follow_wrapper_chains=True,\n",
    "                             skip_bound_arg=True,\n",
    "                             sigcls):\n",
    "\n",
    "    \"Private helper function to get signature for arbitrary callable objects.\"\n",
    "\n",
    "    if not callable(obj):\n",
    "        raise TypeError('{!r} is not a callable object'.format(obj))\n",
    "\n",
    "    if isinstance(obj, types.MethodType):\n",
    "        # In this case we skip the first parameter of the underlying\n",
    "        # function (usually `self` or `cls`).\n",
    "        sig = _signature_from_callable(\n",
    "            obj.__func__,\n",
    "            follow_wrapper_chains=follow_wrapper_chains,\n",
    "            skip_bound_arg=skip_bound_arg,\n",
    "            sigcls=sigcls)\n",
    "\n",
    "        if skip_bound_arg:\n",
    "            return _signature_bound_method(sig)\n",
    "        else:\n",
    "            return sig\n",
    "\n",
    "    # Was this function wrapped by a decorator?\n",
    "    if follow_wrapper_chains:\n",
    "        obj = unwrap(obj, stop=(lambda f: hasattr(f, \"__signature__\")))\n",
    "        if isinstance(obj, types.MethodType):\n",
    "            # If the unwrapped object is a *method*, we might want to\n",
    "            # skip its first parameter (self).\n",
    "            # See test_signature_wrapped_bound_method for details.\n",
    "            return _signature_from_callable(\n",
    "                obj,\n",
    "                follow_wrapper_chains=follow_wrapper_chains,\n",
    "                skip_bound_arg=skip_bound_arg,\n",
    "                sigcls=sigcls)\n",
    "\n",
    "    try:\n",
    "        sig = obj.__signature__\n",
    "    except AttributeError:\n",
    "        pass\n",
    "    else:\n",
    "        if sig is not None:\n",
    "            if not isinstance(sig, Signature):\n",
    "                raise TypeError(\n",
    "                    'unexpected object {!r} in __signature__ '\n",
    "                    'attribute'.format(sig))\n",
    "            return sig\n",
    "\n",
    "    try:\n",
    "        partialmethod = obj._partialmethod\n",
    "    except AttributeError:\n",
    "        pass\n",
    "    else:\n",
    "        if isinstance(partialmethod, functools.partialmethod):\n",
    "            # Unbound partialmethod (see functools.partialmethod)\n",
    "            # This means, that we need to calculate the signature\n",
    "            # as if it's a regular partial object, but taking into\n",
    "            # account that the first positional argument\n",
    "            # (usually `self`, or `cls`) will not be passed\n",
    "            # automatically (as for boundmethods)\n",
    "\n",
    "            wrapped_sig = _signature_from_callable(\n",
    "                partialmethod.func,\n",
    "                follow_wrapper_chains=follow_wrapper_chains,\n",
    "                skip_bound_arg=skip_bound_arg,\n",
    "                sigcls=sigcls)\n",
    "\n",
    "            sig = _signature_get_partial(wrapped_sig, partialmethod, (None,))\n",
    "            first_wrapped_param = tuple(wrapped_sig.parameters.values())[0]\n",
    "            if first_wrapped_param.kind is Parameter.VAR_POSITIONAL:\n",
    "                # First argument of the wrapped callable is `*args`, as in\n",
    "                # `partialmethod(lambda *args)`.\n",
    "                return sig\n",
    "            else:\n",
    "                sig_params = tuple(sig.parameters.values())\n",
    "                assert (not sig_params or\n",
    "                        first_wrapped_param is not sig_params[0])\n",
    "                new_params = (first_wrapped_param,) + sig_params\n",
    "                return sig.replace(parameters=new_params)\n",
    "\n",
    "    if isfunction(obj) or _signature_is_functionlike(obj):\n",
    "        # If it's a pure Python function, or an object that is duck type\n",
    "        # of a Python function (Cython functions, for instance), then:\n",
    "        return _signature_from_function(sigcls, obj)\n",
    "\n",
    "    if _signature_is_builtin(obj):\n",
    "        return _signature_from_builtin(sigcls, obj,\n",
    "                                       skip_bound_arg=skip_bound_arg)\n",
    "\n",
    "    if isinstance(obj, functools.partial):\n",
    "        wrapped_sig = _signature_from_callable(\n",
    "            obj.func,\n",
    "            follow_wrapper_chains=follow_wrapper_chains,\n",
    "            skip_bound_arg=skip_bound_arg,\n",
    "            sigcls=sigcls)\n",
    "        return _signature_get_partial(wrapped_sig, obj)\n",
    "\n",
    "    sig = None\n",
    "    if isinstance(obj, type):\n",
    "        # obj is a class or a metaclass\n",
    "\n",
    "        # First, let's see if it has an overloaded __call__ defined\n",
    "        # in its metaclass\n",
    "        call = _signature_get_user_defined_method(type(obj), '__call__')\n",
    "        if call is not None:\n",
    "            sig = _signature_from_callable(\n",
    "                call,\n",
    "                follow_wrapper_chains=follow_wrapper_chains,\n",
    "                skip_bound_arg=skip_bound_arg,\n",
    "                sigcls=sigcls)\n",
    "        else:\n",
    "            # Now we check if the 'obj' class has a '__new__' method\n",
    "            new = _signature_get_user_defined_method(obj, '__new__')\n",
    "            if new is not None:\n",
    "                sig = _signature_from_callable(\n",
    "                    new,\n",
    "                    follow_wrapper_chains=follow_wrapper_chains,\n",
    "                    skip_bound_arg=skip_bound_arg,\n",
    "                    sigcls=sigcls)\n",
    "            else:\n",
    "                # Finally, we should have at least __init__ implemented\n",
    "                init = _signature_get_user_defined_method(obj, '__init__')\n",
    "                if init is not None:\n",
    "                    sig = _signature_from_callable(\n",
    "                        init,\n",
    "                        follow_wrapper_chains=follow_wrapper_chains,\n",
    "                        skip_bound_arg=skip_bound_arg,\n",
    "                        sigcls=sigcls)\n",
    "\n",
    "        if sig is None:\n",
    "            # At this point we know, that `obj` is a class, with no user-\n",
    "            # defined '__init__', '__new__', or class-level '__call__'\n",
    "\n",
    "            for base in obj.__mro__[:-1]:\n",
    "                # Since '__text_signature__' is implemented as a\n",
    "                # descriptor that extracts text signature from the\n",
    "                # class docstring, if 'obj' is derived from a builtin\n",
    "                # class, its own '__text_signature__' may be 'None'.\n",
    "                # Therefore, we go through the MRO (except the last\n",
    "                # class in there, which is 'object') to find the first\n",
    "                # class with non-empty text signature.\n",
    "                try:\n",
    "                    text_sig = base.__text_signature__\n",
    "                except AttributeError:\n",
    "                    pass\n",
    "                else:\n",
    "                    if text_sig:\n",
    "                        # If 'obj' class has a __text_signature__ attribute:\n",
    "                        # return a signature based on it\n",
    "                        return _signature_fromstr(sigcls, obj, text_sig)\n",
    "\n",
    "            # No '__text_signature__' was found for the 'obj' class.\n",
    "            # Last option is to check if its '__init__' is\n",
    "            # object.__init__ or type.__init__.\n",
    "            if type not in obj.__mro__:\n",
    "                # We have a class (not metaclass), but no user-defined\n",
    "                # __init__ or __new__ for it\n",
    "                if (obj.__init__ is object.__init__ and\n",
    "                    obj.__new__ is object.__new__):\n",
    "                    # Return a signature of 'object' builtin.\n",
    "                    return sigcls.from_callable(object)\n",
    "                else:\n",
    "                    raise ValueError(\n",
    "                        'no signature found for builtin type {!r}'.format(obj))\n",
    "\n",
    "    elif not isinstance(obj, _NonUserDefinedCallables):\n",
    "        # An object with __call__\n",
    "        # We also check that the 'obj' is not an instance of\n",
    "        # _WrapperDescriptor or _MethodWrapper to avoid\n",
    "        # infinite recursion (and even potential segfault)\n",
    "        call = _signature_get_user_defined_method(type(obj), '__call__')\n",
    "        if call is not None:\n",
    "            try:\n",
    "                sig = _signature_from_callable(\n",
    "                    call,\n",
    "                    follow_wrapper_chains=follow_wrapper_chains,\n",
    "                    skip_bound_arg=skip_bound_arg,\n",
    "                    sigcls=sigcls)\n",
    "            except ValueError as ex:\n",
    "                msg = 'no signature found for {!r}'.format(obj)\n",
    "                raise ValueError(msg) from ex\n",
    "\n",
    "    if sig is not None:\n",
    "        # For classes and objects we skip the first parameter of their\n",
    "        # __call__, __new__, or __init__ methods\n",
    "        if skip_bound_arg:\n",
    "            return _signature_bound_method(sig)\n",
    "        else:\n",
    "            return sig\n",
    "\n",
    "    if isinstance(obj, types.BuiltinFunctionType):\n",
    "        # Raise a nicer error message for builtins\n",
    "        msg = 'no signature found for builtin function {!r}'.format(obj)\n",
    "        raise ValueError(msg)\n",
    "\n",
    "    raise ValueError('callable {!r} is not supported by signature'.format(obj))\n",
    "\"\"\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8640cee2",
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "inspect.__dict__"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "04a43d3b",
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "import ast"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e94351dc",
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "block = ast.parse(sigOld, mode='exec')\n",
    "exec(compile(block, '<string>', mode='exec'), globals().update(inspect.__dict__))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "769e3c10",
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "list(locals().keys())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "63c2dbdb",
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "_signature_from_callableOld?"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "50a2ffa6",
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "_signature_from_callable?"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2b01c701",
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "inspect._signature_from_callable = _signature_from_callableOld\n",
    "inspect.signature(Foo) # not a problem for python 3.9+, but is a problem for python 3.7"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a37282c9-6030-47a3-b273-fe073adad482",
   "metadata": {},
   "source": [
    "## Imports"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d105220e-7e7c-4c27-83b0-34030058a298",
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "from fastdebug.core import *"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4aa5804b-8e50-42ba-9007-7949fab3b4ec",
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "import inspect\n",
    "from inspect import *\n",
    "from inspect import _signature_from_callable\n",
    "from inspect import _signature_is_functionlike, _signature_is_builtin, _signature_get_user_defined_method, _signature_from_function, _signature_bound_method"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "eff5129c-8340-4d96-ad12-39708b22f4c0",
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "import fastcore.meta as fm"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8df39492-a712-4985-9bbb-34aea9a2b4dd",
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "_signature_from_callableNew = _signature_from_callable"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6a60d38c-f02d-41ef-9082-c3ed225e2d37",
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "def _signature_from_callableOld(obj, *,\n",
    "                             follow_wrapper_chains=True,\n",
    "                             skip_bound_arg=True,\n",
    "                             sigcls):\n",
    "\n",
    "    \"\"\"Private helper function to get signature for arbitrary\n",
    "    callable objects.\n",
    "    \"\"\"\n",
    "\n",
    "    if not callable(obj):\n",
    "        raise TypeError('{!r} is not a callable object'.format(obj))\n",
    "\n",
    "    if isinstance(obj, types.MethodType):\n",
    "        # In this case we skip the first parameter of the underlying\n",
    "        # function (usually `self` or `cls`).\n",
    "        sig = _signature_from_callable(\n",
    "            obj.__func__,\n",
    "            follow_wrapper_chains=follow_wrapper_chains,\n",
    "            skip_bound_arg=skip_bound_arg,\n",
    "            sigcls=sigcls)\n",
    "\n",
    "        if skip_bound_arg:\n",
    "            return _signature_bound_method(sig)\n",
    "        else:\n",
    "            return sig\n",
    "\n",
    "    # Was this function wrapped by a decorator?\n",
    "    if follow_wrapper_chains:\n",
    "        obj = unwrap(obj, stop=(lambda f: hasattr(f, \"__signature__\")))\n",
    "        if isinstance(obj, types.MethodType):\n",
    "            # If the unwrapped object is a *method*, we might want to\n",
    "            # skip its first parameter (self).\n",
    "            # See test_signature_wrapped_bound_method for details.\n",
    "            return _signature_from_callable(\n",
    "                obj,\n",
    "                follow_wrapper_chains=follow_wrapper_chains,\n",
    "                skip_bound_arg=skip_bound_arg,\n",
    "                sigcls=sigcls)\n",
    "\n",
    "    try:\n",
    "        sig = obj.__signature__\n",
    "    except AttributeError:\n",
    "        pass\n",
    "    else:\n",
    "        if sig is not None:\n",
    "            if not isinstance(sig, Signature):\n",
    "                raise TypeError(\n",
    "                    'unexpected object {!r} in __signature__ '\n",
    "                    'attribute'.format(sig))\n",
    "            return sig\n",
    "\n",
    "    try:\n",
    "        partialmethod = obj._partialmethod\n",
    "    except AttributeError:\n",
    "        pass\n",
    "    else:\n",
    "        if isinstance(partialmethod, functools.partialmethod):\n",
    "            # Unbound partialmethod (see functools.partialmethod)\n",
    "            # This means, that we need to calculate the signature\n",
    "            # as if it's a regular partial object, but taking into\n",
    "            # account that the first positional argument\n",
    "            # (usually `self`, or `cls`) will not be passed\n",
    "            # automatically (as for boundmethods)\n",
    "\n",
    "            wrapped_sig = _signature_from_callable(\n",
    "                partialmethod.func,\n",
    "                follow_wrapper_chains=follow_wrapper_chains,\n",
    "                skip_bound_arg=skip_bound_arg,\n",
    "                sigcls=sigcls)\n",
    "\n",
    "            sig = _signature_get_partial(wrapped_sig, partialmethod, (None,))\n",
    "            first_wrapped_param = tuple(wrapped_sig.parameters.values())[0]\n",
    "            if first_wrapped_param.kind is Parameter.VAR_POSITIONAL:\n",
    "                # First argument of the wrapped callable is `*args`, as in\n",
    "                # `partialmethod(lambda *args)`.\n",
    "                return sig\n",
    "            else:\n",
    "                sig_params = tuple(sig.parameters.values())\n",
    "                assert (not sig_params or\n",
    "                        first_wrapped_param is not sig_params[0])\n",
    "                new_params = (first_wrapped_param,) + sig_params\n",
    "                return sig.replace(parameters=new_params)\n",
    "\n",
    "    if isfunction(obj) or _signature_is_functionlike(obj):\n",
    "        # If it's a pure Python function, or an object that is duck type\n",
    "        # of a Python function (Cython functions, for instance), then:\n",
    "        return _signature_from_function(sigcls, obj)\n",
    "\n",
    "    if _signature_is_builtin(obj):\n",
    "        return _signature_from_builtin(sigcls, obj,\n",
    "                                       skip_bound_arg=skip_bound_arg)\n",
    "\n",
    "    if isinstance(obj, functools.partial):\n",
    "        wrapped_sig = _signature_from_callable(\n",
    "            obj.func,\n",
    "            follow_wrapper_chains=follow_wrapper_chains,\n",
    "            skip_bound_arg=skip_bound_arg,\n",
    "            sigcls=sigcls)\n",
    "        return _signature_get_partial(wrapped_sig, obj)\n",
    "\n",
    "    sig = None\n",
    "    if isinstance(obj, type):\n",
    "        # obj is a class or a metaclass\n",
    "\n",
    "        # First, let's see if it has an overloaded __call__ defined\n",
    "        # in its metaclass\n",
    "        call = _signature_get_user_defined_method(type(obj), '__call__')\n",
    "        if call is not None:\n",
    "            sig = _signature_from_callable(\n",
    "                call,\n",
    "                follow_wrapper_chains=follow_wrapper_chains,\n",
    "                skip_bound_arg=skip_bound_arg,\n",
    "                sigcls=sigcls)\n",
    "        else:\n",
    "            # Now we check if the 'obj' class has a '__new__' method\n",
    "            new = _signature_get_user_defined_method(obj, '__new__')\n",
    "            if new is not None:\n",
    "                sig = _signature_from_callable(\n",
    "                    new,\n",
    "                    follow_wrapper_chains=follow_wrapper_chains,\n",
    "                    skip_bound_arg=skip_bound_arg,\n",
    "                    sigcls=sigcls)\n",
    "            else:\n",
    "                # Finally, we should have at least __init__ implemented\n",
    "                init = _signature_get_user_defined_method(obj, '__init__')\n",
    "                if init is not None:\n",
    "                    sig = _signature_from_callable(\n",
    "                        init,\n",
    "                        follow_wrapper_chains=follow_wrapper_chains,\n",
    "                        skip_bound_arg=skip_bound_arg,\n",
    "                        sigcls=sigcls)\n",
    "\n",
    "        if sig is None:\n",
    "            # At this point we know, that `obj` is a class, with no user-\n",
    "            # defined '__init__', '__new__', or class-level '__call__'\n",
    "\n",
    "            for base in obj.__mro__[:-1]:\n",
    "                # Since '__text_signature__' is implemented as a\n",
    "                # descriptor that extracts text signature from the\n",
    "                # class docstring, if 'obj' is derived from a builtin\n",
    "                # class, its own '__text_signature__' may be 'None'.\n",
    "                # Therefore, we go through the MRO (except the last\n",
    "                # class in there, which is 'object') to find the first\n",
    "                # class with non-empty text signature.\n",
    "                try:\n",
    "                    text_sig = base.__text_signature__\n",
    "                except AttributeError:\n",
    "                    pass\n",
    "                else:\n",
    "                    if text_sig:\n",
    "                        # If 'obj' class has a __text_signature__ attribute:\n",
    "                        # return a signature based on it\n",
    "                        return _signature_fromstr(sigcls, obj, text_sig)\n",
    "\n",
    "            # No '__text_signature__' was found for the 'obj' class.\n",
    "            # Last option is to check if its '__init__' is\n",
    "            # object.__init__ or type.__init__.\n",
    "            if type not in obj.__mro__:\n",
    "                # We have a class (not metaclass), but no user-defined\n",
    "                # __init__ or __new__ for it\n",
    "                if (obj.__init__ is object.__init__ and\n",
    "                    obj.__new__ is object.__new__):\n",
    "                    # Return a signature of 'object' builtin.\n",
    "                    return sigcls.from_callable(object)\n",
    "                else:\n",
    "                    raise ValueError(\n",
    "                        'no signature found for builtin type {!r}'.format(obj))\n",
    "\n",
    "    elif not isinstance(obj, _NonUserDefinedCallables):\n",
    "        # An object with __call__\n",
    "        # We also check that the 'obj' is not an instance of\n",
    "        # _WrapperDescriptor or _MethodWrapper to avoid\n",
    "        # infinite recursion (and even potential segfault)\n",
    "        call = _signature_get_user_defined_method(type(obj), '__call__')\n",
    "        if call is not None:\n",
    "            try:\n",
    "                sig = _signature_from_callable(\n",
    "                    call,\n",
    "                    follow_wrapper_chains=follow_wrapper_chains,\n",
    "                    skip_bound_arg=skip_bound_arg,\n",
    "                    sigcls=sigcls)\n",
    "            except ValueError as ex:\n",
    "                msg = 'no signature found for {!r}'.format(obj)\n",
    "                raise ValueError(msg) from ex\n",
    "\n",
    "    if sig is not None:\n",
    "        # For classes and objects we skip the first parameter of their\n",
    "        # __call__, __new__, or __init__ methods\n",
    "        if skip_bound_arg:\n",
    "            return _signature_bound_method(sig)\n",
    "        else:\n",
    "            return sig\n",
    "\n",
    "    if isinstance(obj, types.BuiltinFunctionType):\n",
    "        # Raise a nicer error message for builtins\n",
    "        msg = 'no signature found for builtin function {!r}'.format(obj)\n",
    "        raise ValueError(msg)\n",
    "\n",
    "    raise ValueError('callable {!r} is not supported by signature'.format(obj))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "77c4c7f2",
   "metadata": {},
   "source": [
    "## Prepare environment variables for debugging"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9e3cc1f5-eff0-48fa-af89-57d8bac4c763",
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "len(dir(fm))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "02872983-d32d-441a-b2c9-72efdb790fe0",
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "len(inspect.__dict__)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d3c367f9-a101-4d65-80be-191ec52faf31",
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "g = {}\n",
    "# g.update(inspect.__dict__)\n",
    "# g.update(fm.__dict__)\n",
    "g.update(inspect._signature_from_callable.__globals__)\n",
    "g.update(fm.delegates.__globals__)\n",
    "\n",
    "len(g)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "463e1a33-b4a2-4b79-a9ab-5366f82e2a36",
   "metadata": {},
   "source": [
    "## When or why to use `FixSigMeta`?"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6a8c8aaa-a78c-434d-858f-f1e564774e6f",
   "metadata": {},
   "source": [
    "When we want a class e.g., `Foo` to have signature from its `__init__` method.\n",
    "\n",
    "`FixSigMeta` can avoid potential problems for `Foo` to access signature from `__init__`."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4c0f9c62-2899-403b-ad95-a177d23764cc",
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "class Foo:\n",
    "    def __init__(self, a, b, c): pass\n",
    "\n",
    "    @classmethod\n",
    "    def clsmed(): pass\n",
    "    \n",
    "inspect.signature(Foo)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "04e81f94-42f7-418a-a9eb-880a185cfaef",
   "metadata": {},
   "source": [
    "## How Foo borrow sig from `__init__`"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "54fef17a-37bb-4027-a77d-7d3c6324ba3a",
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# g = locals()\n",
    "# sig = Fastdb(_signature_from_callable, g)\n",
    "sig = Fastdb(_signature_from_callable)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "86850e17-68ab-4f3f-8962-b74bbc374971",
   "metadata": {},
   "source": [
    "### How to debug `inspect._signature_from_callable` with Fastdb"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "859f5f73-b72d-4367-b4db-41af83e8ed49",
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "#| column: screen\n",
    "\n",
    "sig.dbprint(9, \"so that it can use in itself\")\n",
    "sig.print(part=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "97699555-898d-4545-b997-e8b24b92bbae",
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "#| column: screen\n",
    "\n",
    "sig.dbprint(14, \"obj must be callable\")\n",
    "sig.print(part=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "421b2595-7185-4802-84a1-07c6987d0802",
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "isinstance(Foo.clsmed, types.MethodType)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "98633665-5de6-4c64-a7e8-abee5f206a91",
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "#| column: screen\n",
    "\n",
    "sig.dbprint(17, \"obj can be a classmethod\")\n",
    "sig.print(part=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e4bba8b6-e02a-4ad4-9f6a-ebcb3b330ebe",
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "from fastcore.meta import delegates"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b4a0d1a6-26d7-493c-a16e-25dbca9b2a65",
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "def low(a, b=1): pass\n",
    "@delegates(low)\n",
    "def mid(c, d=1, **kwargs): pass"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "268f375f-8885-4290-8729-8da0a198f512",
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "#| column: screen\n",
    "\n",
    "dbsig = sig.dbprint(28, \"does Foo has __signature__?\", \"follow_wrapper_chains\", \\\n",
    "            \"obj = unwrap(obj, stop=(lambda f: hasattr(f, '__signature__')))\", \"isinstance(obj, types.MethodType)\")\n",
    "\n",
    "# inspect._signature_from_callable = _signature_from_callable\n",
    "inspect._signature_from_callable = dbsig\n",
    "# inspect._signature_from_callable = g['_signature_from_callable']\n",
    "\n",
    "inspect.signature(mid)\n",
    "sig.print(part=1) "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "21917315-9823-42a4-af20-9c268055e1c9",
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "#| column: screen\n",
    "\n",
    "dbsig = sig.dbprint(37, \"check __signature__\", \"obj = unwrap(obj, stop=(lambda f: hasattr(f, '__signature__')))\", \"obj.__signature__\")\n",
    "inspect._signature_from_callable = dbsig\n",
    "inspect.signature(mid)\n",
    "sig.print(part=2) "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "98a7c55f-7d2d-4ff0-81a1-2d5956282fc8",
   "metadata": {},
   "source": [
    "### How exactly Foo get sig from `__init__`"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "38391220-726f-4f1e-8c01-e58bebb74001",
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c1aaf808-6232-465f-a5ef-fe1a2f898c8d",
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "#| column: screen\n",
    "\n",
    "dbsig = sig.dbprint(91, \"step 1: obj is a class?\", \"isinstance(obj, type)\")\n",
    "inspect._signature_from_callable = dbsig\n",
    "inspect.signature(Foo)\n",
    "sig.print(part=3) "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7add02d3-29af-4021-9fc7-dc826ba5f5ef",
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "#| column: screen\n",
    "\n",
    "dbsig = sig.dbprint(96, \"step 2: define its own __call__?\", \"call = _signature_get_user_defined_method(type(obj), '__call__')\")\n",
    "inspect._signature_from_callable = dbsig\n",
    "inspect.signature(Foo)\n",
    "sig.print(part=3) "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "42fbad6c-4fcf-468f-8c55-5f12bee743aa",
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "#| column: screen\n",
    "\n",
    "dbsig = sig.dbprint(101, \"step 3: define its own __new__?\", \"new = _signature_get_user_defined_method(obj, '__new__')\")\n",
    "inspect._signature_from_callable = dbsig\n",
    "inspect.signature(Foo)\n",
    "sig.print(part=4)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "abeeea99-979f-477f-9fe3-002ad1e32953",
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "#| column: screen\n",
    "\n",
    "sig.dbprint(102, \"step 4: define its own __init__?\", \"init = _signature_get_user_defined_method(obj, '__init__')\")\n",
    "inspect._signature_from_callable = dbsig\n",
    "inspect.signature(Foo)\n",
    "sig.print(part=4)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "83eacc2f-9410-4066-bd9c-35b28a37420b",
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "#| column: screen\n",
    "\n",
    "sig.dbprint(108, \"step 5: __init__ is inside obj.__dict__?\", \"'__init__' in obj.__dict__\")\n",
    "inspect._signature_from_callable = dbsig\n",
    "inspect.signature(Foo)\n",
    "sig.print(part=4)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "62d0d5eb-6aa7-4118-8b26-52e1e0a67a23",
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "#| column: screen\n",
    "dbsig = sig.dbprint(116, \"step 6: run on itself using functools.partial\", \"sig = _get_signature_of(factory_method)\")\n",
    "inspect._signature_from_callable = dbsig\n",
    "inspect.signature(Foo)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "574bdf05-a64b-446d-be2a-d0c90b0b12ce",
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "#| column: screen\n",
    "dbsig = sig.dbprint(76, \"step 7: run on itself will run here\", \"isfunction(obj)\", \"_signature_is_functionlike(obj)\")\n",
    "inspect._signature_from_callable = dbsig\n",
    "inspect.signature(Foo.__init__)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b475eec4-28e6-4c4d-ab8c-5629cc1c55fd",
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "#| column: screen\n",
    "dbsig = sig.dbprint(79, \"step 8: get sig with a different func\", \"env\")\n",
    "inspect._signature_from_callable = dbsig\n",
    "inspect.signature(Foo)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "01799b09-51bc-49df-9542-ecbf466bf51f",
   "metadata": {},
   "source": [
    "### Read commented `_signature_from_callable` from python 3.9+"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "852bfe7f-daf9-4dee-8de5-c78d93d43ed4",
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "#| column: screen\n",
    "sig.print()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "768c739b-0c04-4714-85fb-56df01df2371",
   "metadata": {},
   "source": [
    "## Foo's super class overriding `__new__` can stop Foo getting sig from `__init__`"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "aedca067-3237-4981-8620-b9bd379c831f",
   "metadata": {},
   "source": [
    "Many things can go wrong to prevent a class to use the signature from `__init__`. \n",
    "\n",
    "FixSigMeta is a metaclass, which helps us to get our classes' signature right.\n",
    "\n",
    "Then what types of the signature problems can FixSigMeta fix?"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "211c1b73-a877-4e7a-8624-f05f22fd5a41",
   "metadata": {},
   "source": [
    "### When Foo's super class override `__new__`, python3.7 can't give Foo sig from `__init__`"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f84ddcda-98ec-4aeb-b7e0-53005ecd96b0",
   "metadata": {},
   "source": [
    "1. when your class Foo inherits from class Base, if Base defines its `__new__`, then Foo can't get signature from `__init__`. (True for python 3.7 see [demos](https://www.kaggle.com/code/danielliao/notebook3edc928f49?scriptVersionId=104385507&cellId=1), no more for 3.9+)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "63a0bc0e-eb25-44ca-926d-926c67c350f5",
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "class Base: # pass\n",
    "    def __new__(self, **args): pass  # defines a __new__ \n",
    "\n",
    "class Foo(Base):\n",
    "    def __init__(self, d, e, f): pass\n",
    "    \n",
    "inspect._signature_from_callable = sig.orisrc\n",
    "inspect.signature(Foo) # no more problem for python 3.9+, "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9cd349f5-56dc-494c-9ee8-4e280654af6c",
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# from IPython.display import IFrame\n",
    "\n",
    "# IFrame(src=\"https://www.kaggle.com/embed/danielliao/notebook3edc928f49?cellIds=2&kernelSessionId=104407182\", width = \"1200\", height=\"300\", \\\n",
    "#        style=\"margin: 0 auto; width: 100%; max-width: 950px;\", frameborder=\"0\", scrolling=\"auto\", title=\"notebook3edc928f49\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e9c62530-718a-425b-8844-f5bf873e32ca",
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "inspect._signature_from_callable = _signature_from_callableOld\n",
    "inspect.signature(Foo) # it is a problem for python 3.7, "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "98dc4995-65ff-4a0c-b8c7-394c812b9e21",
   "metadata": {},
   "source": [
    "### How python3.7 and its inspect mess it up"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8eac738e-c907-431c-b9cb-80267784285c",
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# g = locals()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0920b881-41e4-4093-91e8-e8e6fe34957d",
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# sigOld = Fastdb(_signature_from_callableOld, g)\n",
    "sigOld = Fastdb(_signature_from_callableOld)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7489d644-905f-44dc-bec1-7b444d657e64",
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "#| column: screen\n",
    "\n",
    "dbsigOld = sigOld.dbprint(115, \"messup step 1: overriding __new__ is detected\", \"new = _signature_get_user_defined_method(obj, '__new__')\")\n",
    "inspect._signature_from_callable = dbsigOld \n",
    "print(inspect.signature(Foo)) # it is a problem for python 3.7, \n",
    "sigOld.print(part=4)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "110127d9",
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "from pprint import pprint"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5c0bf4dc-6ee7-4299-9713-521025d591be",
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "#| column: screen\n",
    "\n",
    "dbsigOld = sigOld.dbprint(117, \"messup step 2: only __new__ sig is extracted\", \"env\")\n",
    "inspect._signature_from_callable = dbsigOld \n",
    "pprint(inspect.signature(Foo)) # it is a problem for python 3.7, \n",
    "sigOld.print(part=4)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a6f037e2-67d5-4bd0-9237-418047a2339e",
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "#| column: screen\n",
    "\n",
    "dbsigOld = sigOld.dbprint(124, \"messup step 3: __init__ don't even get accessed\")\n",
    "inspect._signature_from_callable = dbsigOld \n",
    "pprint(inspect.signature(Foo))\n",
    "sigOld.print(part=4)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d1d25d16-5d91-40c0-b4d7-c877a3bbd38a",
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "edf42128-a9b9-428c-9a0c-d47984f93784",
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "markdown",
   "id": "ad64f258-a7ba-47b7-a410-e9ebe31f0f4b",
   "metadata": {},
   "source": [
    "### FixSigMeta can fix it for python 3.7 inspect"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "99208e7f-fd8c-4a1b-93de-f247af1ff859",
   "metadata": {},
   "source": [
    "Solution to 1: By also inheriting from the metaclass FixSigMeta can solve the signature problem for Foo (for python 3.7)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "252ee17f-0320-4a0a-a454-006bdb27a0ae",
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "from fastcore.meta import FixSigMeta, test_sig"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e9a3e5da-0efe-4b25-8dc2-2c80379f4404",
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "inspect._signature_from_callable = _signature_from_callableNew"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7395b3e8-4e2a-48db-b123-d2ace1c4d679",
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "class Base: # pass\n",
    "    def __new__(self, **args): pass  # defines a __new__ \n",
    "\n",
    "class Foo(Base, metaclass=FixSigMeta):\n",
    "    def __init__(self, d, e, f): pass\n",
    "    \n",
    "test_sig(Foo, '(d, e, f)')\n",
    "inspect.signature(Foo)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "99888a52-4fe0-4243-bd18-c4ebf9033e22",
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "inspect._signature_from_callable = dbsigOld\n",
    "inspect.signature(Foo) # No more a problem for python 3.7"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "04bf7d14-c985-494d-9077-2d425a494e13",
   "metadata": {},
   "source": [
    "### How FixSigMeta fix it?"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f956ca35-9373-4409-a016-bd47637a4398",
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "class Base: # pass\n",
    "    def __new__(self, **args): pass  # defines a __new__ \n",
    "\n",
    "class Foo(Base):\n",
    "    def __init__(self, d, e, f): pass\n",
    "    \n",
    "inspect._signature_from_callable = sigOld.orisrc\n",
    "inspect.signature(Foo) "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "924a54aa-fa6c-4de3-9cca-3c9b0aad7486",
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "hasattr(Foo, '__signature__')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "262d2688-3013-42bc-91ac-512e72bfc25a",
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "class Base: # pass\n",
    "    def __new__(self, **args): pass  # defines a __new__ \n",
    "\n",
    "class Foo(Base, metaclass=FixSigMeta):\n",
    "    def __init__(self, d, e, f): pass\n",
    "    \n",
    "test_sig(Foo, '(d, e, f)')\n",
    "inspect.signature(Foo)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "356b9419-a58c-4280-afa3-b8ad373223d1",
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "Foo.__signature__"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4145b54b-a837-4f85-b0ef-72a9fac27b5b",
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "#| column: screen\n",
    "\n",
    "dbsigOld = sigOld.dbprint(27, \"FixSigMeta step 1: does it have __signature__\", \"env\", \"hasattr(obj, '__signature__')\", \\\n",
    "               \"obj = unwrap(obj, stop=(lambda f: hasattr(f, '__signature__')))\", \"inspect.getdoc(unwrap)\")\n",
    "inspect._signature_from_callable = dbsigOld\n",
    "pprint(inspect.signature(Foo)) \n",
    "sigOld.print(part=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6b6ba2d2-655f-4fc6-9cbd-253cd543206f",
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "#| column: screen\n",
    "\n",
    "dbsigOld = sigOld.dbprint(44, \"FixSigMeta step 2: use __signature__ as Foo's sig\", \"env\", \"sig = obj.__signature__\", \"isinstance(sig, Signature)\")\n",
    "inspect._signature_from_callable = dbsigOld\n",
    "pprint(inspect.signature(Foo)) \n",
    "sigOld.print(part=2)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "117d0557-2c21-438e-a146-23c49d82b712",
   "metadata": {},
   "source": [
    "Note: new and old `_signature_from_callable` have the same code for getting signature for object with `__signature__`."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d0ab1679-9cf0-4c4a-9f7d-d729feedf4a0",
   "metadata": {},
   "source": [
    "### Read commented `_signature_from_callable` of python 3.7"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c5840888-7056-4f88-afe7-a196b9cbbed8",
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "#| column: screen\n",
    "\n",
    "sigOld.print()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2e837887-ec1c-4729-a02e-2698221e3a9f",
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "markdown",
   "id": "7bef5e53-6e0d-438a-8d7f-228e2260a258",
   "metadata": {},
   "source": [
    "## Foo's metaclass defines its own `__call__` will stop Foo get sig from `__init__`"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6401dfb1-d709-45e1-badc-4cee6a18b13d",
   "metadata": {},
   "source": [
    "### Problem demo"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d1c8eded-12a6-4752-b4fc-f0dac6dff9f0",
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "inspect._signature_from_callable = _signature_from_callableNew"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c17d7b71-4932-4685-9f69-a4ee385c684d",
   "metadata": {},
   "source": [
    "2. when your Foo has a metaclass BaseMeta, if BaseMeta need to define its `__call__`, then Foo can't get signature from `__init__`."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "54c934c8-e3ad-4b27-9651-1d29343c74dc",
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "class BaseMeta(type): \n",
    "    # using __new__ from type\n",
    "    def __call__(cls, *args, **kwargs): pass\n",
    "class Foo(metaclass=BaseMeta): \n",
    "    def __init__(self, d, e, f): pass\n",
    "\n",
    "test_sig(Foo, '(*args, **kwargs)')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e405e231-3a0c-48c1-9d61-d82bb09c5663",
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "class BaseMeta(type): \n",
    "    def __new__(cls, name, bases, dict):\n",
    "        return super().__new__(cls, name, bases, dict) # using __new__ from type\n",
    "    def __call__(cls, *args, **kwargs): pass\n",
    "class Foo(metaclass=BaseMeta): \n",
    "    def __init__(self, d, e, f): pass\n",
    "\n",
    "test_sig(Foo, '(*args, **kwargs)')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f7c3d984-8098-4eaa-b4ab-0e668fd33b54",
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "markdown",
   "id": "48413789-b7d6-4a5f-b51c-3928562ed4a8",
   "metadata": {},
   "source": [
    "### Cause of the problem"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "57794a59-99fb-4cc0-aa5e-7095d17552d9",
   "metadata": {},
   "source": [
    "Now I have a better understanding of the source codes, I have 2 places to investigate, they are roughly at line  96 and line 37."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "dabba880-fdff-432c-a6c3-e2560aec46af",
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "#| column: screen\n",
    "\n",
    "dbsig = sig.dbprint(37, \"does Foo store sig inside __signature__\", \"hasattr(obj, '__signature__')\")\n",
    "inspect._signature_from_callable = dbsig\n",
    "pprint(inspect.signature(Foo))\n",
    "sig.print(part=2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "090c4586-e54e-4d22-9a8b-955f8de63ea7",
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "#| column: screen\n",
    "\n",
    "dbsig = sig.dbprint(98, \"__call__ is defined\", \"sig = _get_signature_of(call)\")\n",
    "inspect._signature_from_callable = dbsig\n",
    "pprint(inspect.signature(Foo))\n",
    "sig.print(part=3)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a6c14cc1-4780-4c75-97b7-9a0c95ff8a80",
   "metadata": {},
   "source": [
    "### Solution demo"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "9c051680-f28e-4433-91e1-a4ed6b51333d",
   "metadata": {},
   "source": [
    "Solution to problem 2: you need to inherit from FixSigMeta instead of type when constructing the metaclass to preserve the signature in `__init__`. Be careful not to override `__new__` when doing this:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "93b4401b-0370-4a12-9165-cf77ebf0edad",
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "class BaseMeta(FixSigMeta): \n",
    "    # using __new__ of  FixSigMeta instead of type\n",
    "    def __call__(cls, *args, **kwargs): pass\n",
    "\n",
    "class Foo(metaclass=BaseMeta): # Base\n",
    "    def __init__(self, d, e, f): pass\n",
    "\n",
    "test_sig(Foo, '(d, e, f)')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "562e6e5c-f2e3-477a-9968-e4faaa8f5116",
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "class BaseMeta(FixSigMeta): \n",
    "    def __new__(cls, name, bases, dict): # not really overriding __new__, still using FixSigMeta.__new__ actually\n",
    "        return super().__new__(cls, name, bases, dict)\n",
    "    def __call__(cls, *args, **kwargs): pass\n",
    "\n",
    "class Foo(metaclass=BaseMeta): # Base\n",
    "    def __init__(self, d, e, f): pass\n",
    "\n",
    "test_sig(Foo, '(d, e, f)')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "40e4d5cb-39d2-404d-ac4a-b6b3e3b850ab",
   "metadata": {},
   "source": [
    "Note: if Base also defines `__new__`, then FixSigMeta can't help. "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "29b255c9-e874-4e93-b312-c1632d9c7cb0",
   "metadata": {},
   "source": [
    "### How FixSigMeta fix this problem"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5292209d-3131-4b34-9973-7ce0618d660f",
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "#| column: screen\n",
    "\n",
    "dbsig = sig.dbprint(29, \"why has to unwrap?\", \"hasattr(obj, '__signature__')\")\n",
    "inspect._signature_from_callable = dbsig\n",
    "pprint(inspect.signature(Foo))\n",
    "sig.print(part=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "980fb8aa-4c09-4efb-921a-1a7c9e26014c",
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "#| column: screen\n",
    "\n",
    "dbsig = sig.dbprint(30, \"what is wrapped by Foo?\", \"isinstance(obj, types.MethodType)\", \"obj\", \"type(obj)\")\n",
    "inspect._signature_from_callable = dbsig\n",
    "pprint(inspect.signature(Foo))\n",
    "sig.print(part=2)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "030aca0d-3801-4285-8cb1-64b965be4e31",
   "metadata": {},
   "source": [
    "### Common feature of the solutions above by FixSigMeta"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "28c4c1fb-22cc-4377-a26d-ba2ff3087617",
   "metadata": {},
   "source": [
    "The key is to create `__signature__` for Foo, so that `inspect.signature` will get sig from `__signature__`, instead of `__new__` or `__call__`"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8172be99",
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "from fastcore.meta import FixSigMeta"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4dcbca6a",
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "import fastcore.meta as fm"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "48f1aeca",
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "fm.__dict__"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e3c7acb4",
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "eval('fastcore.meta')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "41c3892f",
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "delegates.__dict__"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3728213f-58e3-45df-8851-be7b68acf537",
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# g = locals()\n",
    "# fsm = Fastdb(FixSigMeta, g)\n",
    "fsm = Fastdb(FixSigMeta)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d07e1661-3f54-48b2-bb40-542436dfcfb7",
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "from fastcore.meta import _rm_self"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d55af8be",
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "#| column: screen\n",
    "\n",
    "dbfsm = fsm.dbprint(4, \"FixSigMeta create Foo with its __new__\", \"res\", \"inspect.signature(res.__init__)\", \"_rm_self(inspect.signature(res.__init__))\")\n",
    "FixSigMeta = dbfsm\n",
    "inspect._signature_from_callable = sig.orisrc # deactivate it\n",
    "\n",
    "class BaseMeta(FixSigMeta): \n",
    "    def __new__(cls, name, bases, dict): # not really overriding __new__, still using FixSigMeta.__new__ actually\n",
    "        return super().__new__(cls, name, bases, dict)\n",
    "    def __call__(cls, *args, **kwargs): pass\n",
    "\n",
    "class Foo(metaclass=BaseMeta): # Base\n",
    "    def __init__(self, d, e, f): pass\n",
    "\n",
    "test_sig(Foo, '(d, e, f)')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ca57a717-f091-4024-a16d-e2dde27a4653",
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "fsm.print()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "30f10214",
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "FixSigMeta = fsm.orisrc"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "466e91f9-f59e-41f6-8c91-4982e24fa9c9",
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "from fastcore.meta import test_eq"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2b0c96ce-2274-45e7-bac4-3ce12709f064",
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "class BaseMeta(FixSigMeta): \n",
    "    # __new__ comes from FixSigMeta\n",
    "    def __new__(cls, *args, **kwargs): pass # as it create None for Foo, there is no signature neither\n",
    "    def __call__(cls, *args, **kwargs): pass\n",
    "\n",
    "class Foo(metaclass=BaseMeta): # Base\n",
    "    def __init__(self, d, e, f): pass\n",
    "\n",
    "test_eq(type(Foo), type(None))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "bdc52175-d2e5-4324-8f3b-8e7b86d34ab8",
   "metadata": {},
   "source": [
    "Note: if Base also defines `__init__`, then FixSigMeta can still help. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1a2049e9-eba6-444c-afc0-cb7aea5372c7",
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "class BaseMeta(FixSigMeta): \n",
    "    # __new__ comes from FixSigMeta\n",
    "    def __init__(cls, *args, **kwargs): pass # this __init__ is not used by Foo\n",
    "    def __call__(cls, *args, **kwargs): pass\n",
    "\n",
    "class Foo(metaclass=BaseMeta): # Base\n",
    "    def __init__(self, d, e, f): pass # override the __init__ above\n",
    "\n",
    "test_sig(Foo, '(d, e, f)')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "da6452ad-ca23-4e16-9610-0844a22160bb",
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "hide_input": false,
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.13"
  },
  "toc": {
   "base_numbering": 1,
   "nav_menu": {},
   "number_sections": true,
   "sideBar": false,
   "skip_h1_title": false,
   "title_cell": "Table of Contents",
   "title_sidebar": "Contents",
   "toc_cell": false,
   "toc_position": {
    "height": "244px",
    "left": "1152px",
    "top": "92px",
    "width": "288px"
   },
   "toc_section_display": true,
   "toc_window_display": true
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}