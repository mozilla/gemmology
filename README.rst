.. code::

                                          _
                                         | |
      __ _  ___ _ __ ___  _ __ ___   ___ | | ___   __ _ _   _
     / _` |/ _ \ '_ ` _ \| '_ ` _ \ / _ \| |/ _ \ / _` | | | |
    | (_| |  __/ | | | | | | | | | | (_) | | (_) | (_| | |_| |
     \__, |\___|_| |_| |_|_| |_| |_|\___/|_|\___/ \__, |\__, |
      __/ |                                        __/ | __/ |
     |___/                                        |___/ |___/

                                                    version 0.1

Small Integer Matrix Multiply
=============================

Gemmology is a rewrite of `intgemm <https://github.com/kpu/intgemm>`_ with a focus
on 8-bit integer matrix multiplication and using
`xsimd <https://github.com/QuantStack/xsimd>`_ as an abstract vector instructrion
set when possible.

THe original algorithm and API are left mostly untouched, appart from a few
namespace changes.

Usage
-----

Gemmology consists in a single header file, just drop it in your project to use
it, then mostly follow `intgemm`_ API:

.. code:: c++

   #include "gemmology.h"

    float alpha = 25;
    float quant_mult = 127/alpha;
    gemmology::PrepareA(A, A_prepared, quant_mult, A_rows, width);
    gemmology::PrepareB(B, B_prepared, quant_mult, width, B_cols);

    /* Prepare the bias (inplace) */
    float unquant_mult_forprep = (-1)*(alpha)*(alpha)/(127.0f);
    gemmology::Shift::PrepareBias(B_prepared, width, B_cols,
                                  callbacks::UnquantizeAndAddBiasAndWrite(unquant_mult_forprep, inputBias, inputBias));
    /* Multiply */
    gemmology::Shift::Multiply(A_prepared, B_prepared, A_rows, width, B_cols,
                               callbacks::UnquantizeAndAddBiasAndWrite(unquant_mult_forprep, bias, C));

Difference with `intgemm`_
--------------------------

Gemmology only handles quantized matrix of 8-bit integers.

Gemmology provides an SSE2 implementation of the original algorithm, while
`intgemm`_ stops at SSSE3. The SSE2 version is
roughly 2.5 slower than the SSSE3 version.

All Gemmology functions are parametrized by a target architecture (e.g.
``xsimd::sse4_2``) which is set to the best available at compile time. It's up
to the user to handle the dynamic dispatch (eventually using `xsimd generic
mechanism <https://xsimd.readthedocs.io/en/latest/api/dispatching.html>`_ to do so.

Testing
-------

All tests lie in the ``test`` directory, a sample test invocation (provided
`xsimd`_ and `sde64
<https://www.intel.fr/content/www/fr/fr/download/684897/intel-software-development-emulator.html>`_
are available on your system.

.. code::

   make -C test XSIMD_INCLUDE_DIR=~/source/xsimd/include SDE64=~/Downloads/sde-external-9.14.0-2022-10-25-lin/sde64

Acknowledgments
---------------

This is really mostly a portage of `intgemm`_ to `xsimd`_. So big thanks to
`intgemm`_ authors for the original work.
