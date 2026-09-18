# syntax=docker/dockerfile:1.7

ARG SGLANG_IMAGE=lmsysorg/sglang@sha256:687efca081e85f4e3126456ff389b1af515fc08a604de4c61f947f531963aba7
ARG FLASHINFER_CACHE_IMAGE=hongccc/sglang-omni@sha256:374d0b1c30b2bff685b1716fc64a02ad3b3d0a90fe2ce73ce9861a6992c28101

FROM ${FLASHINFER_CACHE_IMAGE} AS flashinfer-cache
FROM ${SGLANG_IMAGE} AS runtime

ARG UCX_COMMIT=d8e50df6651b9ea5b76f23aee0aefbf053a4137a

RUN git clone --filter=blob:none https://github.com/openucx/ucx.git /tmp/ucx \
    && git -C /tmp/ucx checkout "${UCX_COMMIT}" \
    && cd /tmp/ucx \
    && ./autogen.sh \
    && ./contrib/configure-release-mt \
        --enable-shared \
        --disable-static \
        --disable-doxygen-doc \
        --enable-optimizations \
        --enable-cma \
        --enable-devel-headers \
        --with-cuda=/usr/local/cuda \
        --with-verbs \
        --with-dm \
        --prefix=/usr/local \
    && make -j"$(nproc)" \
    && make install-strip \
    && ldconfig \
    && rm -rf /tmp/ucx

# Install the release lock and the source from this build context. Runtime
# startup must not clone/pull a mutable branch or replace the MOSS fork.
COPY deployment/repro/requirements.lock /tmp/requirements.lock
COPY deployment/repro/build-constraints.txt /tmp/build-constraints.txt
RUN uv pip install --system --break-system-packages --require-hashes \
        --index-url https://pypi.org/simple --build-constraint /tmp/build-constraints.txt -r /tmp/requirements.lock \
    && python3 -m pip uninstall -y flashinfer-cubin flashinfer-jit-cache \
    && rm /tmp/requirements.lock

COPY pyproject.toml README.md LICENSE /opt/sglang-omni/
COPY sglang_omni /opt/sglang-omni/sglang_omni
COPY sglang_omni_router /opt/sglang-omni/sglang_omni_router
COPY examples /opt/sglang-omni/examples
COPY deployment /opt/sglang-omni/deployment
RUN uv pip install --system --break-system-packages --no-deps --no-build-isolation -e /opt/sglang-omni

# Docker builds do not have a GPU to regenerate architecture-specific JIT
# artifacts. Reuse the Python 3.12 cache validated on Ada (SM89) and Hopper
# (SM90a); other architectures compile into their own cache directory.
COPY --from=flashinfer-cache /root/.cache/flashinfer/0.6.14 /root/.cache/flashinfer/0.6.14

ENV FLASHINFER_WORKSPACE_BASE=/root \
    FLASHINFER_JIT_DEBUG=0

RUN <<'EOF'
cat >/usr/local/bin/sglang-omni-entrypoint <<'SCRIPT'
#!/usr/bin/env bash
set -euo pipefail

cd /opt/sglang-omni
cuda_home="$(python3 deployment/repro/cuda_toolkit.py)"
test -x "$cuda_home/bin/nvcc"
export CUDA_HOME="$cuda_home" CUDA_PATH="$cuda_home"
export PATH="$cuda_home/bin:$PATH"
exec /opt/nvidia/nvidia_entrypoint.sh "$@"
SCRIPT
chmod 0755 /usr/local/bin/sglang-omni-entrypoint
EOF

WORKDIR /opt/sglang-omni
ENTRYPOINT ["/usr/local/bin/sglang-omni-entrypoint"]
CMD ["/bin/zsh"]
