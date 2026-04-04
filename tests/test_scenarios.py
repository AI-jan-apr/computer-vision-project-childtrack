# اختبار كل سيناريو من الـ 9 سيناريوهات
# التأكد إن كل حالة تطلع النتيجة الصحيحة

# السناريوهات:

# سيناريوهات طبيعية:
# 1- بالغ + طفل يدخلون ويخرجون معاً ← حالة طبيعية 
# 2- بالغين + طفلين يدخلون ويخرجون معاً ← مجموعة كاملة 

# سيناريوهات تنبيه:

# 3- بالغ + طفل يدخلون، الطفل يخرج وحده ← تنبيه
# 4- بالغ + طفل يدخلون، البالغ يخرج وحده ← طفل منسي
# 5- طفل يدخل مع مجموعة A ويخرج مع مجموعة B ← مشبوه

# سيناريوهات اختبار اللوجيك:

# 6- مجموعتان يدخلون قريب من بعض لكن ليسوا مجموعة واحدة ← اختبار Distance
# 7- زحمة عند الباب ← اختبار Detection
# 8- شخص يحجب شخص آخر (Occlusion) ← اختبار Tracking
# 9- شخص يلحق بمجموعة بعد انتهاء الـ 4 ثواني ← اختبار Time Window
from group_manager import GroupManager
from alert_engine import AlertEngine
import time

from matcher import Matcher


def fake(tid, label, direction, cx=100, cy=200):
    return {
        'track_id': tid,
        'label': label,
        'direction': direction,
        'centroid': (cx, cy),
        'bbox': [0, 0, 10, 10],
        'conf': 0.9
    }


gm = GroupManager()
ae = AlertEngine()
matcher = Matcher()

print("\n--- SCENARIO: Adult leaves child ---")

# entry
gm.update([
    fake(1, 'adult', 'entry', 100, 200),
    fake(2, 'child', 'entry', 110, 200)
])

# close window
time.sleep(4.2)
gm.update([
    fake(1, 'adult', 'stationary'),
    fake(2, 'child', 'stationary')
])

# adult exits
for _ in range(8):
    events = gm.update([
        fake(1, 'adult', 'exit'),
        fake(2, 'child', 'stationary')
    ])

alerts = ae.process(events)

print("\nALERTS:")
for a in alerts:
    print(a)


print("\n--- SCENARIO: Child exits alone ---")

gm = GroupManager()
ae = AlertEngine()

gm.update([
    fake(1, 'adult', 'entry', 100, 200),
    fake(2, 'child', 'entry', 110, 200)
])

time.sleep(4.2)
gm.update([
    fake(1, 'adult', 'stationary'),
    fake(2, 'child', 'stationary')
])

# adult exits first
for _ in range(8):
    gm.update([
        fake(1, 'adult', 'exit'),
        fake(2, 'child', 'stationary')
    ])

# child exits alone
for _ in range(8):
    events = gm.update([
        fake(2, 'child', 'exit')
    ])

alerts = ae.process(events)

print("\nALERTS:")
for a in alerts:
    print(a)