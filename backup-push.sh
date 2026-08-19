source .env
set -euo pipefail

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
DB_PATH="${DATA_DIR_PATH}database/database.db"
BACKUP_PATH="${DATA_DIR_PATH}database/database-${TIMESTAMP}.db"

#
# Create local backup via VACUUM INTO (safe with concurrent WAL writers)
#
echo "[$(date)] Starting database backup via VACUUM INTO..."

ionice -c 3 nice -n 19 .venv/bin/python -c "
import sqlite3, sys, time
src, dst = sys.argv[1], sys.argv[2]
start = time.time()
conn = sqlite3.connect(src)
conn.execute('PRAGMA busy_timeout = 300000')
conn.execute(\"VACUUM INTO '\" + dst + \"'\")
conn.close()
print(f'VACUUM INTO completed in {(time.time() - start) / 60:.1f} minutes')
" "${DB_PATH}" "${BACKUP_PATH}"

#
# Verify backup integrity before uploading
#
echo "[$(date)] Verifying backup integrity..."

.venv/bin/python -c "
import sqlite3, sys
conn = sqlite3.connect(sys.argv[1])
result = conn.execute('PRAGMA quick_check(1)').fetchone()
conn.close()
if result[0] != 'ok':
    print(f'INTEGRITY CHECK FAILED: {result[0]}', file=sys.stderr)
    sys.exit(1)
print('Integrity check passed')
" "${BACKUP_PATH}"

#
# Reclaim WAL space accumulated during backup
#
.venv/bin/python -c "
import sqlite3, sys
conn = sqlite3.connect(sys.argv[1])
conn.execute('PRAGMA wal_checkpoint(PASSIVE)')
conn.close()
" "${DB_PATH}"

echo "[$(date)] Backup size: $(du -h "${BACKUP_PATH}" | cut -f1)"

#
# Push to R2 via rclone
#
echo "[$(date)] Uploading backup to R2..."

rclone copyto "${BACKUP_PATH}" "${BACKUP_RCLONE_REMOTE}:${BACKUP_RCLONE_BUCKET}/database/database-${TIMESTAMP}.db" --transfers 1 --checkers 1 --multi-thread-streams 4 --s3-upload-concurrency 4 --s3-chunk-size 64M -v --s3-no-check-bucket;
rclone copy "${DATA_DIR_PATH}logs" "${BACKUP_RCLONE_REMOTE}:${BACKUP_RCLONE_BUCKET}/logs" --transfers 16 --checkers 16 --multi-thread-streams 16 --s3-upload-concurrency 16 -v;
rclone copy "${DATA_DIR_PATH}dashboard" "${BACKUP_RCLONE_REMOTE}:${BACKUP_RCLONE_BUCKET}/dashboard" --transfers 16 --checkers 16 --multi-thread-streams 16 --s3-upload-concurrency 16 -v;

#
# Keep only the 3 most recent local backups
#
#echo "[$(date)] Cleaning up old local backups..."
#ls -t "${DATA_DIR_PATH}database/database-"*.db 2>/dev/null | tail -n +4 | xargs rm -f 2>/dev/null || true

#echo "[$(date)] Backup complete."
