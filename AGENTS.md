# AGENTS.md

## Adding a New Strategy

ทุกครั้งที่เพิ่ม strategy ใหม่ ต้องทำทุกข้อด้านล่างให้ครบก่อนถือว่างานส่งมอบเสร็จ:

1. เพิ่ม strategy เข้า precompute โดยใช้ evaluation period และ alignment rules เดียวกับ strategy เดิมใน family เดียวกัน
2. เพิ่ม standalone latest-weight refresh path ที่คำนวณ latest weights ใหม่จาก data/cache ปัจจุบันของ repo นี้ ห้ามพึ่ง static latest-weight files จาก `dynamic_port_opt` สำหรับ deployed runtime behavior
3. เพิ่ม latest-weight refresh script และ output files เข้า daily GitHub Action ที่ใช้ update latest weights
4. เพิ่มคำอธิบาย strategy แบบ user-facing ใน app/docs เป็นภาษาไทย โดยใส่ strategy settings, universe, optimizer/model, rebalance/timing rules, daily exposure rules, caps และ latest weights มาจากไหน คำ technical เช่น optimizer, rebalance, signal, exposure, drawdown, cache ใช้ภาษาอังกฤษหรือทับศัพท์ได้ถ้าอ่านง่ายกว่า
5. ทำให้ latest-weight display ซ่อน asset rows ที่ portfolio weight ต่ำกว่า `1%` สำหรับทุก strategy ยกเว้น user ขอ inspect small residual positions โดยตรง ต้องใช้ rule นี้ให้สม่ำเสมอทั้ง strategy ใหม่และ strategy เดิม

## Strategy Description Style

เมื่อเพิ่มหรือแก้คำอธิบาย strategy แบบ user-facing ใน app/docs:

1. เขียนคำอธิบายเป็น bullet sections ที่อ่านง่าย ไม่เขียนเป็น paragraph ยาวก้อนเดียว
2. แยกคำอธิบายอย่างน้อยเป็น:
   - `Strategy setup`
   - `Daily exposure rules`
3. ใน `Strategy setup` ต้องมี base allocation หรือ sleeve mix, universe, selection rules, optimizer/model, objective, rebalance schedule, caps และ latest-weight source
4. ใน `Daily exposure rules` ต้องบอกชัดว่าใช้ daily exposure หรือไม่ ถ้าใช้ ต้องมี:
   - signal timing โดยเฉพาะ lag-1 หรือ next-session execution
   - signal และ threshold ของแต่ละ asset/sleeve
   - เมื่อ signal เป็น risk-off แล้ว exposure เหลือเท่าไร
   - reduced exposure ไปอยู่ cash, BIL, sleeve อื่น หรือ `Cash / Reduced Exposure`
5. ห้าม duplicate คำอธิบายเดียวกันทั้งใน gray caption และ info box ให้ใช้ bullet info box เป็นหลักสำหรับรายละเอียด user-facing
6. สำหรับ strategy ที่ไม่มี overlay ต้องเขียนให้ชัดว่าไม่มี daily exposure overlay และ sleeves ยัง active จนถึง rebalance ถัดไป

## Chart Display Rules

1. Chart ที่แสดง performance ของ strategy ต้องแสดงข้อมูลเต็มเท่าที่ strategy นั้นมีให้มากที่สุดเป็น default ห้ามตัด history ทั้ง dataset ให้สั้นลงเพียงเพราะมี strategy บางตัวที่ข้อมูลสั้นกว่า
2. ถ้า user เลือกเปรียบเทียบ strategy หลายตัวใน chart เดียวกัน ให้ปรับช่วงเวลาให้เท่ากันเฉพาะคู่หรือชุด strategy ที่กำลังแสดงอยู่เท่านั้น
3. ถ้า strategy ที่เลือกมีข้อมูลน้อยกว่า เช่น strategy ที่มี Japan PIT history สั้นกว่า ให้ตัด chart ของ strategy อื่นใน chart เดียวกันให้ตรงกับช่วงเวลาของ strategy ที่ข้อมูลสั้นกว่า เพื่อให้ metrics และ visual comparison อยู่บน window เดียวกัน
4. การตัดช่วงเวลาเพื่อ comparison ต้องไม่เปลี่ยน precomputed return history ต้นฉบับของ strategy อื่น และต้องไม่ทำให้ strategy ที่ไม่ได้ถูกเลือกใน chart สูญเสีย history เต็มของตัวเอง

## SET100 Updates

ใช้ `https://www.set.or.th/api/set/index/set100/composition?lang=th` เป็น source URL เมื่อ update latest SET100 composition
