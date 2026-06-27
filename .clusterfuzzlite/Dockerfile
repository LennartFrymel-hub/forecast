# SPDX-FileCopyrightText: 2026 Lennart Frymel
# SPDX-License-Identifier: AGPL-3.0-or-later

FROM gcr.io/oss-fuzz-base/base-builder-python

COPY . $SRC/forecast
COPY .clusterfuzzlite/build.sh $SRC/build.sh
RUN chmod +x $SRC/build.sh

WORKDIR $SRC/forecast
