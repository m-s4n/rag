Yeni belge geldiğinde yalnızca şunu yapman yeterli:

python build_index.py

tam rag için yerel llm ekle
şuanda sistem yalnızca doğrudan alıntı yapıyor.
buna yerel llm eklersek artık o alıntılardan akıcı türkçe  özet üretir.

(Bunun için ollama run llama3.1:8b + llm_answer_with_ollama() fonksiyonu ekleyeceğiz.)