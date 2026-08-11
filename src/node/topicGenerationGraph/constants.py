import os

MODEL = os.getenv("PRACTICE_GENERATION_MODEL", "gpt-5.4")
DUPLICATE_THRESHOLD = 0.90

# So vong de xuat toi da trong MOT luot sinh.
#
# Vong 1 de xuat tu do (prompt khong kem danh sach kho). Neu bo loc trung cat mat mot so, vong sau
# duoc goi lai KEM TEN chu de vua va cham -- thong tin cu the va dung luc, thay cho viec doc truoc
# ca kho roi tu nho.
#
# 3 la tran chan tren, khong phai so vong thuong chay: vong lap dung ngay khi da du so chu de can
# hoac khong con va cham nao. Dat cao hon chi keo dai them do tre cho hoc sinh dang ngoi cho o
# duong synchronousOffers.
#
# Cung mo hinh voi EditorNode cua do thi sinh CAU HOI (MAX_EDITOR_ROUNDS): sua lai chi nhung cai
# truot, co tran vong.
MAX_PROPOSAL_ROUNDS = 3
