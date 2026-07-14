# AGENTS.md

## Adding a New Strategy

ทุกครั้งที่เพิ่ม strategy ใหม่ ต้องทำทุกข้อด้านล่างให้ครบก่อนถือว่างานส่งมอบเสร็จ:

1. เพิ่ม strategy เข้า precompute โดยใช้ evaluation period และ alignment rules เดียวกับ strategy เดิมใน family เดียวกัน
2. เพิ่ม standalone latest-weight refresh path ที่คำนวณ latest weights ใหม่จาก data/cache ปัจจุบันของ repo นี้ ห้ามพึ่ง static latest-weight files จาก `dynamic_port_opt` สำหรับ deployed runtime behavior
3. เพิ่ม latest-weight refresh script และ output files เข้า daily GitHub Action ที่ใช้ update latest weights
4. เพิ่มคำอธิบาย strategy แบบ user-facing ใน app/docs เป็นภาษาไทย โดยใส่ strategy settings, universe, optimizer/model, rebalance/timing rules, daily exposure rules, caps และ latest weights มาจากไหน คำ technical เช่น optimizer, rebalance, signal, exposure, drawdown, cache ใช้ภาษาอังกฤษหรือทับศัพท์ได้ถ้าอ่านง่ายกว่า
5. ทำให้ latest-weight display ซ่อน asset rows ที่ portfolio weight ต่ำกว่า `1%` สำหรับทุก strategy ยกเว้น user ขอ inspect small residual positions โดยตรง ต้องใช้ rule นี้ให้สม่ำเสมอทั้ง strategy ใหม่และ strategy เดิม

## Replacing an Existing Strategy

เมื่อ replace strategy เดิมด้วย strategy ใหม่ ต้องทำให้ครบทั้ง backend behavior และ user-facing label ก่อนถือว่างานเสร็จ:

1. เปลี่ยน backend series/source/precompute ให้ชี้ไป strategy ใหม่จริง
2. เปลี่ยน dropdown/display label หรือ config key ที่ user เห็นใน app ให้เป็นชื่อ strategy ใหม่ ห้ามเหลือ label เก่าถ้า user ตั้งใจให้ replace ไม่ใช่เพิ่ม alias
3. เปลี่ยน active/default strategy list ที่ใช้สร้าง options ให้ใช้ชื่อใหม่ และตรวจว่า state/default เก่าไม่ดึง label เดิมกลับมา
4. เปลี่ยน latest-weight file, metadata file, latest_weights_strategy และ output path ให้ตรงกับ strategy ใหม่
5. เปลี่ยน user-facing description/settings/docs ให้ตรงกับ strategy ใหม่ รวม Strategy setup และ Daily exposure rules ตามกฎด้านล่าง
6. ถ้าจำเป็นต้องรองรับ state เก่า ให้ทำ migration/alias แบบมองไม่เห็น user แต่อย่าให้ dropdown แสดงชื่อ strategy เก่า
7. smoke test หลังแก้ โดย import app หรือรัน check ที่เทียบเท่า แล้ว assert ว่า dropdown options ไม่มีชื่อ strategy เก่า และมีชื่อ strategy ใหม่

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

## Language and Encoding Rules

1. ไฟล์ source, config, docs และ generated text ที่มีภาษาไทยต้องบันทึกเป็น UTF-8 และต้องแสดงภาษาไทยได้ถูกต้อง ห้ามมี mojibake เช่น `Ã`, `Â`, `â€` หรือข้อความภาษาไทยที่กลายเป็น `???`
2. ห้ามส่งข้อความภาษาไทยผ่าน shell pipeline หรือเครื่องมือที่อาจแปลง encoding โดยไม่ตรวจผลลัพธ์ หากต้องสร้างหรือแก้ข้อความผ่าน script ต้องใช้ UTF-8 แบบ explicit และอ่านไฟล์กลับมาตรวจหลังเขียน
3. ทุกครั้งที่เพิ่มหรือแก้ user-facing text ให้รัน smoke test ที่ import/parse app หรือ config แล้ว assert ว่าข้อความที่เกี่ยวข้องมีอักขระไทยจริง และไม่มี mojibake markers หรือ `???`
4. ต้องตรวจทั้งข้อความใน app, docs, metadata, precomputed/static dashboard export และ generated JSON ที่ user มองเห็น ไม่ตรวจเฉพาะ source file จุดเดียว
5. ถ้าพบ mojibake เดิมใน section หรือ strategy ที่กำลังแก้ ต้องแก้ให้หมดใน scope เดียวกันก่อนถือว่างานเสร็จ โดยห้ามเปลี่ยน backend behavior เพียงเพื่อแก้ภาษา

## Chart Display Rules

1. Chart ที่แสดง performance ของ strategy ต้องแสดงข้อมูลเต็มเท่าที่ strategy นั้นมีให้มากที่สุดเป็น default ห้ามตัด history ทั้ง dataset ให้สั้นลงเพียงเพราะมี strategy บางตัวที่ข้อมูลสั้นกว่า
2. ถ้า user เลือกเปรียบเทียบ strategy หลายตัวใน chart เดียวกัน ให้ปรับช่วงเวลาให้เท่ากันเฉพาะคู่หรือชุด strategy ที่กำลังแสดงอยู่เท่านั้น
3. ถ้า strategy ที่เลือกมีข้อมูลน้อยกว่า เช่น strategy ที่มี Japan PIT history สั้นกว่า ให้ตัด chart ของ strategy อื่นใน chart เดียวกันให้ตรงกับช่วงเวลาของ strategy ที่ข้อมูลสั้นกว่า เพื่อให้ metrics และ visual comparison อยู่บน window เดียวกัน
4. การตัดช่วงเวลาเพื่อ comparison ต้องไม่เปลี่ยน precomputed return history ต้นฉบับของ strategy อื่น และต้องไม่ทำให้ strategy ที่ไม่ได้ถูกเลือกใน chart สูญเสีย history เต็มของตัวเอง

## SET100 Updates

ใช้ `https://www.set.or.th/api/set/index/set100/composition?lang=th` เป็น source URL เมื่อ update latest SET100 composition
