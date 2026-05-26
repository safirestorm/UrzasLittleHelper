# UrzasLittleHelper
Velkommen til vores machine learning projekt til eksamen i juni 2026. 

API er tilgængelig på: http://188.166.7.17:8000/docs

## Sådan starter du projektet på din egen computer:
* Clone projektet ned til din computer.
* Kør "Docker compose up -d" i terminalen for at starte en Qdrant server på din lokale maskine.
* Når Qdrant serveren kører skal du loade et "snapShot". Dette kan downloades på: https://drive.google.com/file/d/1DRq8U4glHDw8ROXhRqjVyZC8rEcTU4mc/view?usp=share_link
* Med snapshottet downloaded, kør følgende kommando i terminalen, der hvor snapshottet ligger:
  * curl -X POST http://localhost:6333/collections/mtg_cards/snapshots/upload \ -H "Content-Type: multipart/form-data" \ -F "snapshot=@mtg_cards.snapshot"
* Du kan nu starte API'en ved at køre: "fastapi dev API.py --host 0.0.0.0" i terminalen i projektet.
* For at tilgå funktionerne kan du åbne api'en i browseren med http://0.0.0.0:8000/docs
