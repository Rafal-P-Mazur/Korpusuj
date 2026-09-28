# Third-party notices

Korpusuj includes or uses third-party software and linguistic resources. This file identifies components that require a specific attribution or notice.

The license of Korpusuj is provided separately in `LICENSE.txt`. That file contains the complete text of the GNU General Public License, version 3. Because Herference is also distributed under GPL-3.0, the same complete license text applies and is not duplicated in a separate `COPYING` file.

## Morfeusz 2 and SGJP inflectional data

Korpusuj uses Morfeusz 2 and SGJP inflectional data in the optional lemma-repair workflow.

Copyright holder of Morfeusz 2:

**Institute of Computer Science, Polish Academy of Sciences (Instytut Podstaw Informatyki PAN).**

Authors and copyright holders of SGJP inflectional data:

**Zygmunt Saloni, Włodzimierz Gruszczyński, Marcin Woliński, Robert Wołosz and Danuta Skowrońska.**

Morfeusz 2 and the included linguistic data are distributed under the BSD 2-Clause License:

> All rights reserved.
>
> Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
>
> 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
> 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
>
> THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES, INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION, HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT, INCLUDING NEGLIGENCE OR OTHERWISE, ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Official license page: <https://morfeusz.sgjp.pl/doc/license/en>

Official project page: <https://morfeusz.sgjp.pl/en>

Recommended citation:

> Witold Kieraś, Marcin Woliński. *Morfeusz 2 – analizator i generator fleksyjny dla języka polskiego*. Język Polski, XCVII(1):75–83, 2017.

## Herference

Korpusuj optionally uses Herference as a Polish coreference-resolution component integrated with spaCy.

Project: <https://github.com/ipipan/herference>

Version used by the current Korpusuj environments: **0.2.0**.

Author and project information:

**Karol Saputa, Institute of Computer Science, Polish Academy of Sciences (Instytut Podstaw Informatyki PAN).**

License: **GNU General Public License, version 3.0.**

The complete GPL-3.0 license text is already supplied in `LICENSE.txt`, which is also the license file of Korpusuj. A second identical license file named `COPYING` is therefore not included.

Recommended citation:

> Karol Saputa. 2022. *Coreference Resolution for Polish: Improvements within the CRAC 2022 Shared Task*. In Proceedings of the CRAC 2022 Shared Task on Multilingual Coreference Resolution, pages 18–22. Association for Computational Linguistics.

Herference is used through the spaCy integration. External language-model files are not identified here as components bundled with Korpusuj. Any model obtained separately remains subject to the terms published with that model.
