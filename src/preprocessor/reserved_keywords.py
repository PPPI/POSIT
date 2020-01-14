# >>> import keyword
# >>> print(keyword.kwlist)
python_reserved = ['and', 'as', 'assert', 'break', 'class', 'continue', 'def', 'del', 'elif', 'else', 'except', 'exec',
                   'finally', 'for', 'from', 'global', 'if', 'import', 'in', 'is', 'lambda', 'not', 'or', 'pass',
                   'print', 'raise', 'return', 'try', 'while', 'with', 'yield']
# http://www.jwrider.com/riderist/java/javaidrs.htm
java_reserved = ['abstract', 'assert', 'boolean', 'break', 'byte', 'case', 'catch', 'char', 'class', 'const',
                 'continue', 'default', 'do', 'double', 'else', 'enum', 'extends', 'false', 'final', 'finally', 'float',
                 'for', 'goto', 'if', 'implements', 'import', 'instanceof', 'int', 'interface', 'long', 'native', 'new',
                 'null', 'package', 'private', 'protected', 'public', 'return', 'short', 'static', 'strictfp', 'super',
                 'switch', 'synchronized', 'this', 'throw', 'throws', 'transient', 'true', 'try', 'void', 'volatile',
                 'while']
# http://www.javascripter.net/faq/reserved.htm
javascript_reserved = ['abstract', 'else', 'instanceof', 'super', 'boolean', 'enum', 'int', 'switch', 'break', 'export',
                       'interface', 'synchronized', 'byte', 'extends', 'let', 'this', 'case', 'false', 'long', 'throw',
                       'catch', 'final', 'native', 'throws', 'char', 'finally', 'new', 'transient', 'class', 'float',
                       'null', 'true', 'const', 'for', 'package', 'try', 'continue', 'function', 'private', 'typeof',
                       'debugger', 'goto', 'protected', 'var', 'default', 'if', 'public', 'void', 'delete',
                       'implements', 'return', 'volatile', 'do', 'import', 'short', 'while', 'double', 'in', 'static',
                       'with']
# http://www.javascripter.net/faq/reserved.htm
javascript_implementation_dependent = ['alert', 'frames', 'outerHeight', 'all', 'frameRate', 'outerWidth', 'anchor',
                                       'function', 'packages', 'anchors', 'getClass', 'pageXOffset', 'area',
                                       'hasOwnProperty', 'pageYOffset', 'Array', 'hidden', 'parent', 'assign',
                                       'history', 'parseFloat', 'blur', 'image', 'parseInt', 'button', 'images',
                                       'password', 'checkbox', 'Infinity', 'pkcs11', 'clearInterval', 'isFinite',
                                       'plugin', 'clearTimeout', 'isNaN', 'prompt', 'clientInformation',
                                       'isPrototypeOf', 'propertyIsEnum', 'close', 'java', 'prototype', 'closed',
                                       'JavaArray', 'radio', 'confirm', 'JavaClass', 'reset', 'constructor',
                                       'JavaObject', 'screenX', 'crypto', 'JavaPackage', 'screenY', 'Date',
                                       'innerHeight', 'scroll', 'decodeURI', 'innerWidth', 'secure',
                                       'decodeURIComponent', 'layer', 'select', 'defaultStatus', 'layers', 'self',
                                       'document', 'length', 'setInterval', 'element', 'link', 'setTimeout', 'elements',
                                       'location', 'status', 'embed', 'Math', 'String', 'embeds', 'mimeTypes', 'submit',
                                       'encodeURI', 'name', 'taint', 'encodeURIComponent', 'NaN', 'text', 'escape',
                                       'navigate', 'textarea', 'eval', 'navigator', 'top', 'event', 'Number',
                                       'toString', 'fileUpload', 'Object', 'undefined', 'focus', 'offscreenBuffering',
                                       'unescape', 'form', 'open', 'untaint', 'forms', 'opener', 'valueOf', 'frame',
                                       'option', 'window']
# http://www.javascripter.net/faq/reserved.htm
javascript_eventhandlers = ['onbeforeunload', 'ondragdrop', 'onkeyup', 'onmouseover', 'onblur', 'onerror', 'onload',
                            'onmouseup', 'ondragdrop', 'onfocus', 'onmousedown', 'onreset', 'onclick', 'onkeydown',
                            'onmousemove', 'onsubmit', 'oncontextmenu', 'onkeypress', 'onmouseout', 'onunload']
# http://en.cppreference.com/w/c/keyword
c_reserved = ['auto', 'break', 'case', 'char', 'const', 'continue', 'default', 'do', 'double', 'else', 'enum', 'extern',
              'float', 'for', 'goto', 'if', 'inline', 'int', 'long', 'register', 'restrict', 'return', 'short',
              'signed', 'sizeof', 'static', 'struct', 'switch', 'typedef', 'union', 'unsigned', 'void', 'volatile',
              'while', '_Alignas', '_Alignof', '_Atomic', '_Bool', '_Complex', '_Generic', '_Imaginary', '_Noreturn',
              '_Static_assert', '_Thread_local', 'alignas', 'alignof', 'bool', 'atomic_bool', 'atomic_char',
              'atomic_schar', 'atomic_uchar', 'atomic_short', 'atomic_ushort', 'atomic_int', 'atomic_uint',
              'atomic_long', 'atomic_ulong', 'atomic_llong', 'atomic_ullong', 'atomic_char16_t', 'atomic_char32_t',
              'atomic_wchar_t', 'atomic_int_least8_t', 'atomic_uint_least8_t', 'atomic_int_least16_t',
              'atomic_uint_least16_t', 'atomic_int_least32_t', 'atomic_uint_least32_t', 'atomic_int_least64_t',
              'atomic_uint_least64_t', 'atomic_int_fast8_t', 'atomic_uint_fast8_t', 'atomic_int_fast16_t',
              'atomic_uint_fast16_t', 'atomic_int_fast32_t', 'atomic_uint_fast32_t', 'atomic_int_fast64_t',
              'atomic_uint_fast64_t', 'atomic_intptr_t', 'atomic_uintptr_t', 'atomic_size_t', 'atomic_ptrdiff_t',
              'atomic_intmax_t', 'atomic_uintmax_t', 'complex', 'imaginary', 'noreturn', 'static_assert',
              'thread_local', 'if', 'elif', 'else', 'endif', 'defined', 'ifdef', 'ifndef', 'define', 'undef', 'include',
              'line', 'error', 'pragma', '_Pragma', 'asm', 'fortran']
# http://en.cppreference.com/w/cpp/keyword
cpp_reserved = ['alignas', 'alignof', 'and', 'and_eq', 'asm', 'atomic_cancel', 'atomic_commit', 'atomic_noexcept',
                'auto', 'bitand', 'bitor', 'bool', 'break', 'case', 'catch', 'char', 'char16_t', 'char32_t', 'class',
                'compl', 'concept', 'const', 'constexpr', 'const_cast', 'continue', 'decltype', 'default', 'delete',
                'do', 'double', 'dynamic_cast', 'else', 'enum', 'explicit', 'export', 'extern', 'false', 'float', 'for',
                'friend', 'goto', 'if', 'import', 'inline', 'int', 'long', 'module', 'mutable', 'namespace', 'new',
                'noexcept', 'not', 'not_eq', 'nullptr', 'operator', 'or', 'or_eq', 'private', 'protected', 'public',
                'register', 'reinterpret_cast', 'requires', 'return', 'short', 'signed', 'sizeof', 'static',
                'static_assert', 'static_cast', 'struct', 'switch', 'synchronized', 'template', 'this', 'thread_local',
                'throw', 'true', 'try', 'typedef', 'typeid', 'typename', 'union', 'unsigned', 'using', 'virtual',
                'void', 'volatile', 'wchar_t', 'while', 'xor', 'xor_eq', 'override', 'final', 'transaction_safe',
                'transaction_safe_dynamic', 'if', 'elif', 'else', 'endif', 'defined', 'ifdef', 'ifndef', 'define',
                'undef', 'include', 'line', 'error', 'pragma', '_Pragma']
