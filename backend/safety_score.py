import json

def load_zones():
    with open("data/chennai_zones.json", "r") as f:
        return json.load(f)

def get_safety_score(area_name):
    data = load_zones()
    for zone in data["zones"]:
        if any(area_name.lower() in a.lower() for a in zone["areas"]):
            crime = zone["crime_rate"]
            safety = zone["safety"]
            
            if safety == "green":
                score = 90 - (crime * 2)
                label = "✅ Safe"
            elif safety == "yellow":
                score = 60 - (crime * 2)
                label = "⚠️ Moderate Risk"
            else:
                score = 30 - (crime * 2)
                label = "🔴 High Risk"
            
            return {
                "area": area_name,
                "zone": zone["name"],
                "score": score,
                "label": label,
                "crime_rate": crime
            }
    return {"error": "Area not found"}

if __name__ == "__main__":
    print(get_safety_score("Adyar"))
    print(get_safety_score("Manali"))
    print(get_safety_score("Egmore"))