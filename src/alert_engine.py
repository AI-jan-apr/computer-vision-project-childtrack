# طفل خرج وحده
# بالغ خرج بدون طفل
# طفل خرج مع مجموعة مختلفة
# المجموعة ما خرجت بعد وقت طويل
# خروج سليم = لا تنبيه
# هنا راح نفصل في الكيسس ونحدد متى نطلق تنبيه ومتى لا
import logging

logger = logging.getLogger(__name__)


class AlertEngine:
    def __init__(self):
        pass

    def process(self, exit_events):
        alerts = []

        for e in exit_events:
            group_id = e['group_id']

            # 🔴 CASE 1 — Adult left, child still inside
            if (
                e['exited_label'] == 'adult'
                and len(e['remained_children']) > 0
            ):
                alerts.append({
                    'type': 'CHILD_LEFT_ALONE',
                    'group_id': group_id,
                    'child_ids': e['remained_children'],
                    'adult_left': e['exited_id']
                })

                logger.warning(f"[ALERT] Child left alone in {group_id}")

            # 🔴 CASE 2 — Child exits without adult
            elif (
                e['exited_label'] == 'child'
                and len(e['remained_adults']) == 0
            ):
                alerts.append({
                    'type': 'CHILD_EXITED_ALONE',
                    'group_id': group_id,
                    'child_id': e['exited_id']
                })

                logger.warning(f"[ALERT] Child exited alone in {group_id}")

            # 🟢 CASE 3 — Safe exit
            elif e['all_adults_exited'] and not e['remained_children']:
                logger.info(f"[SAFE] Group {group_id} exited safely")

        return alerts