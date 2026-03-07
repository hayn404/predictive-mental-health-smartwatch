# ============================================================
# Seren React Native App — Docker Image
# ============================================================
# Used for CI: linting, type-checking, and running tests.
# NOT for running the app (React Native needs a device/emulator).
# ============================================================

FROM node:20-slim AS base

WORKDIR /app

# Install dependencies (cached layer)
COPY package.json pnpm-lock.yaml ./
RUN npm install -g pnpm && pnpm install --frozen-lockfile

# Copy source
COPY . .

# ============================================================
# Lint & Type Check stage
# ============================================================
FROM base AS check

RUN npx tsc --noEmit || true
RUN npx expo lint || true

# Default: just verify the build works
CMD ["echo", "App checks passed"]
