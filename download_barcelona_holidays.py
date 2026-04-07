import requests
import pandas as pd
from icalendar import Calendar
from pathlib import Path

ICS_URL = "https://opendata-ajuntament.barcelona.cat/data/dataset/bef03e00-942b-443d-b2e6-d060f5b03cc3/resource/b91875ad-5a94-4c84-8b78-0fe822e53d1a/download"
OUT_CSV = "barcelona_holidays.csv"

def download_barcelona_holidays_csv(out_csv: str = OUT_CSV) -> str:
    resp = requests.get(ICS_URL, timeout=30)
    resp.raise_for_status()

    cal = Calendar.from_ical(resp.content)

    rows = []
    for component in cal.walk():
        if component.name != "VEVENT":
            continue

        dtstart = component.get("DTSTART")
        summary = component.get("SUMMARY")

        if dtstart is None:
            continue

        value = dtstart.dt
        # Soporta date o datetime
        date_value = value.isoformat() if hasattr(value, "isoformat") else str(value)

        rows.append({
            "date": date_value[:10],
            "holiday_name": str(summary) if summary else None,
            "city": "Barcelona",
            "country": "Spain",
            "source": "Open Data BCN"
        })

    df = pd.DataFrame(rows).drop_duplicates().sort_values("date")
    df.to_csv(out_csv, index=False, encoding="utf-8")

    return out_csv


if __name__ == "__main__":
    path = download_barcelona_holidays_csv()
    print(f"CSV guardado en: {Path(path).resolve()}")