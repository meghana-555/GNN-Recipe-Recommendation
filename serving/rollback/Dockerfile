# Assisted by Claude
FROM alpine:3.19
RUN apk add --no-cache bash
COPY rollback.sh /rollback.sh
RUN chmod +x /rollback.sh
CMD ["/rollback.sh"]
