from sentence_transformers import CrossEncoder
from alqac_utils import bm25_tokenizer

model = CrossEncoder('saved_models/mmarco-mMiniLMv2-L12-H384-v1-VN-LegalQA-bm25-512-10')
# scores = model.predict([('Query', 'Paragraph1'), ('Query', 'Paragraph2') , ('Query', 'Paragraph3')])

question = "Trường hợp nào sau đây ngôn ngữ được sử dụng trong tố tụng không phải là tiếng Việt?"

p1 = "Ngôn ngữ\n\n1. Đối với tranh chấp không có yếu tố nước ngoài, ngôn ngữ sử dụng trong tố tụng trọng\n\ntài là tiếng Việt, trừ trường hợp tranh chấp mà ít nhất một bên là doanh nghiệp có vốn đầu tư nước ngoài. Trường hợp bên tranh chấp không sử dụng được tiếng Việt thì được chọn người phiên dịch ra tiếng Việt.\n\n2. Đối với tranh chấp có yếu tố nước ngoài, tranh chấp mà ít nhất một bên là doanh nghiệp có vốn đầu tư nước ngoài, ngôn ngữ sử dụng trong tố tụng trọng tài do các bên thỏa thuận. Trường hợp các bên không có thỏa thuận thì ngôn ngữ sử dụng trong tố tụng trọng tài do Hội đồng trọng tài quyết định." 

p2 = "Tiếng nói và chữ viết dùng trong tố tụng hành chính\n\nTiếng nói và chữ viết dùng trong tố tụng hành chính là tiếng Việt.\n\nNgười tham gia tố tụng hành chính có quyền dùng tiếng nói và chữ viết của dân tộc mình; trường hợp này phải có người phiên dịch.\n\nNgười tham gia tố tụng hành chính là người khuyết tật nghe, người khuyết tật nói hoặc người khuyết tật nhìn có quyền dùng ngôn ngữ, ký hiệu, chữ dành riêng cho người khuyết tật; trường hợp này phải có người biết nghe, nói bằng ngôn ngữ, ký hiệu, chữ dành riêng của người khuyết tật để dịch lại."

cleaned_question = ' '.join(bm25_tokenizer(question))
cleaned_p1 = ' '.join(bm25_tokenizer(p1))
cleaned_p2 = ' '.join(bm25_tokenizer(p2))

print (f'cleaned_question: {cleaned_question}')
print (f'cleaned_p1: {cleaned_p1}')
print (f'cleaned_p2: {cleaned_p2}')
print (model.predict([(cleaned_question, cleaned_p1), (cleaned_question, cleaned_p2)]))