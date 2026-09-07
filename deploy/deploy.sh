#!/usr/bin/env bash
#
# Deploy hhy.homes to the VPS (SvelteKit Node adapter).
#
# ── 前置（已在本机完成）────────────────────────────────────────────────────
#   pnpm add -D @sveltejs/adapter-node && 改 svelte.config.js 为 adapter-node
#   pnpm build   # 产物是自包含的 build/（含 static/），服务器只需 Node 运行时
#
# ── 用法 ──────────────────────────────────────────────────────────────────
#   ./deploy/deploy.sh                 # 用 deploy key（~/.ssh/hhy_homes，推荐）
#   SSH_PASS='你的密码' ./deploy/deploy.sh   # 用密码（本机需装 sshpass）
#
# 密码只从 SSH_PASS 环境变量读取，绝不写入任何文件。
set -euo pipefail

SERVER_HOST="124.222.217.211"
SERVER_USER="ubuntu"
DOMAIN="hhy.homes"
REMOTE_DIR="/var/www/hhy.homes"
SSH_HOST="${SERVER_USER}@${SERVER_HOST}"

cd "$(dirname "$0")/.."

# ── 1. 本地构建 ───────────────────────────────────────────────────────────
if [ ! -d build ]; then
  echo "ERROR: 找不到 build/。请先把 svelte.config.js 换成 @sveltejs/adapter-node，再运行 pnpm build。"
  exit 1
fi
echo "==> 本地构建 (Node adapter，剥离 LOCAL_ADMIN，防止编辑界面泄漏进生产)"
env -u LOCAL_ADMIN pnpm build

# ── 2. SSH/rsync 传输方式 ─────────────────────────────────────────────────
# 优先用 deploy key；没有 key 时回退到密码（SSH_PASS，需 sshpass）。
SSH_KEY="${SSH_KEY:-$HOME/.ssh/hhy_homes}"
SSH_OPTS=(-o StrictHostKeyChecking=accept-new)
if [ -f "$SSH_KEY" ]; then
  SSH_OPTS+=(-i "$SSH_KEY")
elif [ -n "${SSH_PASS:-}" ]; then
  command -v sshpass >/dev/null || { echo "需要 sshpass（brew install sshpass 或用 SSH 密钥）"; exit 1; }
  SSH_CMD="sshpass -p $SSH_PASS"
fi

remote() {
  if [ -n "${SSH_CMD:-}" ]; then
    $SSH_CMD ssh "${SSH_OPTS[@]}" "$SSH_HOST" "$@"
  else
    ssh "${SSH_OPTS[@]}" "$SSH_HOST" "$@"
  fi
}

upload() { # $1 = local path (dir), $2 = remote dir
  if [ -n "${SSH_CMD:-}" ]; then
    $SSH_CMD rsync -az "${SSH_OPTS[@]}" "$1" "$SSH_HOST:$2"
  else
    rsync -az -e "ssh ${SSH_OPTS[*]}" "$1" "$SSH_HOST:$2"
  fi
}

echo "==> 准备远程目录"
remote "sudo mkdir -p $REMOTE_DIR && sudo chown -R $SERVER_USER:$SERVER_USER $REMOTE_DIR"

echo "==> 准备评论数据目录与 .env（首次生成管理密钥，仅打印一次）"
remote "mkdir -p $REMOTE_DIR/data && chown -R $SERVER_USER:$SERVER_USER $REMOTE_DIR/data"
remote "test -f $REMOTE_DIR/.env || { ADMIN_KEY=\"\$(head -c 32 /dev/urandom | base64 | tr -dc 'A-Za-z0-9' | head -c 32)\"; SECRET=\"\$(head -c 32 /dev/urandom | base64 | tr -dc 'A-Za-z0-9' | head -c 32)\"; printf 'COMMENTS_ADMIN_KEY=%s\\nSESSION_SECRET=%s\\nCOMMENTS_FILE=%s\\nOWNER_GITHUB_LOGIN=CraigKEttleman\\nSITE_ORIGIN=https://hhy.homes\\n' \"\$ADMIN_KEY\" \"\$SECRET\" \"$REMOTE_DIR/data/comments.json\" > $REMOTE_DIR/.env; echo; echo '=== 评论后台管理密钥（请保存）==='; echo \"COMMENTS_ADMIN_KEY=\$ADMIN_KEY\"; echo '================================='; }"

echo "==> 上传 build/（adapter-node 已把 static/ 打进 build/client）"
upload "build/" "$REMOTE_DIR/build/"

# ── 3. 安装 Node 20+（官方 nodesource 源，装到 /usr/bin/node）──────────────
NODE_BIN="$(remote 'bash -lc "command -v node" 2>/dev/null' || true)"
if [ -n "$NODE_BIN" ]; then
  echo "==> 服务器已有 Node：$NODE_BIN"
else
  echo "==> 安装 Node 20 (nodesource)"
  remote "curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash - && sudo apt-get install -y nodejs"
  NODE_BIN="/usr/bin/node"
fi

# ── 4. 安装 systemd 服务（把 node 路径写进 unit）────────────────────────────
echo "==> 安装 systemd 服务 hhy-homes"
sed "s|__NODE_BIN__|$NODE_BIN|g" deploy/hhy-homes.service > /tmp/hhy-homes.service
upload "/tmp/hhy-homes.service" "/tmp/hhy-homes.service"
remote "sudo mv /tmp/hhy-homes.service /etc/systemd/system/hhy-homes.service && sudo systemctl daemon-reload && sudo systemctl enable hhy-homes && sudo systemctl restart hhy-homes && sleep 1 && systemctl --no-pager status hhy-homes | head -5"

# ── 5. nginx 反代（如已装会跳过安装）────────────────────────────────────────
echo "==> 配置 nginx 反代"
upload "deploy/nginx.conf" "/tmp/hhy-homes-nginx.conf"
remote "command -v nginx >/dev/null || sudo apt-get install -y nginx
sudo cp /tmp/hhy-homes-nginx.conf /etc/nginx/sites-available/hhy.homes
sudo ln -sf /etc/nginx/sites-available/hhy.homes /etc/nginx/sites-enabled/hhy.homes
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t && sudo systemctl reload nginx"

echo ""
echo "✅ 部署完成：https://$DOMAIN"
echo "  评论后台：https://$DOMAIN/admin （密钥见上方打印的 COMMENTS_ADMIN_KEY）"
