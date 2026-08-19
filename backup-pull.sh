source .env

rclone copy "${BACKUP_RCLONE_REMOTE}:${BACKUP_RCLONE_BUCKET}/database" "${DATA_DIR_PATH}database" --transfers 16 --checkers 16 --multi-thread-streams 16 -v; 
rclone copy "${BACKUP_RCLONE_REMOTE}:${BACKUP_RCLONE_BUCKET}/logs" "${DATA_DIR_PATH}logs" --transfers 16 --checkers 16 --multi-thread-streams 16 -v;
rclone copy "${BACKUP_RCLONE_REMOTE}:${BACKUP_RCLONE_BUCKET}/dashboard" "${DATA_DIR_PATH}dashboard" --transfers 16 --checkers 16 --multi-thread-streams 16 -v;
# Add more folders as needed