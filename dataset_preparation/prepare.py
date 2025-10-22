import pandas as pd
import random
from faker import Faker

hasar_onarimlari = pd.read_csv("./hasar_onarimlari.csv")
hasar_turleri = pd.read_csv("./hasar_turleri.csv")
aircraft_models = pd.read_csv("./aircraft_models.csv")

fake = Faker()
aircraft_models_arr = aircraft_models["Uçak Modeli"].tolist()
operators = [fake.company() for _ in range(30)]
cities = [fake.city() for _ in range(30)]
severities = ["Minor", "Moderate", "Major", "Critical"]


def generate_report_text(
    date,
    operator,
    aircraft_type,
    location,
    damage_type,
    severity,
    affected_part,
    repair,
):
    return (
        f"{date.strftime('%d %B %Y')} tarihinde {operator} işletmecisine ait {aircraft_type} tipi uçak, {location} üzerinde {damage_type.lower()} "
        f"olayı yaşamıştır. Bu olay {severity} seviyesi bir olay olarak işaretlenmiştir. "
        f"İncelemede {affected_part.lower()} bölgesinde hasar tespit edilmiştir. "
        f"Gerekli bakım işlemleri tamamlanarak uçak yeniden hizmete alınmıştır."
    )


def generate_report(n=100):
    data = []
    for i in range(n):
        selected_damage_type = hasar_turleri.sample(1).iloc[0]
        selected_repair = hasar_onarimlari[
            hasar_onarimlari["Kod"] == selected_damage_type["Kod"]
        ]

        aircraft_type = random.choice(aircraft_models_arr)
        operator = random.choice(operators)
        location = random.choice(cities)
        severity = random.choice(severities)
        damage_type = selected_damage_type["Hasar Türü"]
        affected_part = selected_damage_type["Etkilenen Parça (affected_part)"]
        date = fake.date_between(start_date="-5y", end_date="today")
        repair = selected_repair["Onarım Açıklaması"].iloc[0]

        data.append(
            {
                "report_id": f"D{i + 1:06d}",
                "aircraft_type": aircraft_type,
                "operator": operator,
                "location": location,
                "severity": severity,
                "damage_type": damage_type,
                "affected_part": affected_part,
                "date": date,
                "repair": repair,
                "report": generate_report_text(
                    date,
                    operator,
                    aircraft_type,
                    location,
                    damage_type,
                    severity,
                    affected_part,
                    repair,
                ),
            }
        )
    return pd.DataFrame(data)


df = generate_report(100000)
df.to_csv(
    "./aircraft_reports.csv",
    index=False,
)
df.head()
