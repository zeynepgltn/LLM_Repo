import json

class Tokenizer:  # Sınıf adı büyük harfle başlamalı
    def __init__(self, vocab_file):
        with open(vocab_file, 'r') as f:
            self.vocab = json.load(f)
            self.reverse_vocab = {v: k for k, v in self.vocab.items()}
    
    def encode(self, text):
        tokens = []
        
        # Metni kelimelere ayır
        for word in text.split():
            i = 0
            
            # Her kelimeyi karakter karakter işle
            while i < len(word):
                found_match = False
                
                # En uzun eşleşmeyi bulmak için geriye doğru ara
                for j in range(len(word), i, -1):
                    sub_word = word[i:j]
                    
                    # Alt kelime vocabularyda var mı kontrol et
                    if sub_word in self.vocab:
                        tokens.append(self.vocab[sub_word])
                        i = j
                        found_match = True
                        break
                
                # Eğer hiçbir eşleşme bulunamazsa, bilinmeyen token ekle
                if not found_match:
                    # <unk> token'ı yoksa varsayılan bir değer kullan
                    unk_token = self.vocab.get("<unk>", 0)
                    tokens.append(unk_token)
                    i += 1
        
        return tokens  # Son token'ı kaldırma işlemini kaldırdım

    def decode(self, tokens):  # Parametre adını düzelttim
        text = []
        for token_id in tokens:  # Değişken adını düzelttim
            if token_id in self.reverse_vocab:
                text.append(self.reverse_vocab[token_id])
            else:
                text.append("<unk>")  # Bilinmeyen token için
        return " ".join(text)  # join kullanarak daha temiz birleştirme