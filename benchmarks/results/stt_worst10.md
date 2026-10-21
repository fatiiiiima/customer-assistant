# Worst clips per engine (condition: office noise)

Ranked by per-clip WER (uncapped). `norm` rows are what the scorer actually compares; `nums` are the canonical numbers it extracted, so a numeric miss can be judged as audio vs scorer. Non-speech probe excluded.

## openvino-whisper:whisper-tiny-int8-ov@CPU+prompt

**num06** (scripted) WER 26.7
- ref : under thirty five thousand and less than sixty thousand kilometres
- hyp : And that that that that that that that that that that that that that that that that that that that that that that that that that that that that that that that that that that that that that that that that that that that t
- norm ref: under thirty five thousand and less than sixty thousand kilometres
- norm hyp: and that that that that that that that that that that that that that that that that that that that that that that that that that that that that that that that that that that that that that that that that that that that t
- nums ref ['35000', '60000'] vs hyp [] -> MISS

**spont029** (spont) WER 16.375
- ref : no no no i don't want this one
- hyp : No, no, no, no, no, no, no, no, no, no, no, no, no, no, no, no, no, no, no, no, no, no, no, no, no, no, no, no, no, no, no, no, no, no, no, no, no, no, no, no, no, no, no, no, no, no, no, no, no, no, no, no, no, no, no, 
- norm ref: no no no i don't want this one
- norm hyp: no no no no no no no no no no no no no no no no no no no no no no no no no no no no no no no no no no no no no no no no no no no no no no no no no no no no no no no no no no no no no no no no no no no no no no no no no n

**phone03** (scripted) WER 1.538
- ref : plus nine seven one five zero four four four three three two two
- hyp : Bless 911, Bless 911, I was 0445, 2322.
- norm ref: plus nine seven one five zero four four four three three two two
- norm hyp: bless nine hundred and eleven bless nine hundred and eleven i was four hundred and forty five two thousand three hundred and twenty two
- nums ref ['971504443322'] vs hyp ['911', '911', '447322'] -> MISS

**spont035** (spont) WER 1.286
- ref : other car rooms are offering cheaper alternatives
- hyp : Buokay, Williams, I love feeling cheaper, I don't eat this
- norm ref: other car rooms are offering cheaper alternatives
- norm hyp: buokay williams i love feeling cheaper i don't eat this

**num03** (scripted) WER 1.0
- ref : can you do forty eight thousand
- hyp : World in your 240-8000.
- norm ref: can you do forty eight thousand
- norm hyp: world in your two hundred and forty eight thousand
- nums ref ['48000'] vs hyp ['248000'] -> MISS

**plate02** (scripted) WER 1.0
- ref : plate number Sharjah three four five six seven
- hyp : Vietnam, Bosia, Jatri, 4567
- norm ref: plate number sharjah three four five six seven
- norm hyp: vietnam bosia jatri four thousand five hundred and sixty seven
- nums ref ['34567'] vs hyp ['4567'] -> MISS

**date02** (scripted) WER 1.0
- ref : book me for tomorrow at ten in the morning
- hyp : Abu Dhabi, Abu Dhabi, Abu Dhabi,
- norm ref: book me for tomorrow at ten in the morning
- norm hyp: abu dhabi abu dhabi abu dhabi
- nums ref ['10'] vs hyp [] -> MISS

**name02** (scripted) WER 1.0
- ref : this is Fatima Al Hashimi calling
- hyp : Nekoyun,
- norm ref: this is fatima al hashimi calling
- norm hyp: nekoyun

**name03** (scripted) WER 1.0
- ref : it is Abdullah Khalid Rashid
- hyp : Eres Abdullah, Fahil Adran, Sheikh
- norm ref: it is abdullah khalid rashid
- norm hyp: eres abdullah fahil adran sheikh

**spont002** (spont) WER 1.0
- ref : the weather is really hot today
- hyp : So whether it is a daily heart today
- norm ref: the weather is really hot today
- norm hyp: so whether it is a daily heart today

### openvino-whisper:whisper-tiny-int8-ov@CPU+prompt: all 14 numeric misses (scripted)

| id | ref numbers | hyp numbers | lenient digit-stream ref / hyp | lenient match | hyp text |
|---|---|---|---|---|---|
| num02 | ['87500'] | ['8500'] | 87500 / 8000500 | no | Look out, has 8,000, 500 kilometers |
| num03 | ['48000'] | ['248000'] | 48000 / 2408000 | no | World in your 240-8000. |
| num06 | ['35000', '60000'] | [] | 3500060000 /  | no | And that that that that that that that that that that that that that that that that that that that t |
| num07 | ['72500'] | ['72000'] | 72500 / 72000 | no | My offer is 72,000, very limited ones |
| num08 | ['2019', '40000'] | ['2019', '1000'] | 201940000 / 20191000 | no | 2019, Nordrubo, Ford, Ford, 1000, kilometers |
| plate01 | ['12345'] | ['9840'] | 12345 / 81840 | no | The Create is the by 8, 1840 slides |
| plate02 | ['34567'] | ['4567'] | 34567 / 4567 | no | Vietnam, Bosia, Jatri, 4567 |
| plate03 | ['30'] | [] | 128901 /  | no | Abu Dhabi, Tviles, Eid, Nanguzi, Deira, |
| phone01 | ['0501234567'] | ['0501034671'] | 0501234567 / 0501034671 | no | My number is 0-5-0-1-0-3-4-6-7-1 |
| phone02 | ['0559876543'] | ['0', '97648'] | 0559876543 / 097648 | no | Follow me, I'm 0, I'm 9, 7, 6, 4, 8 |
| phone03 | ['971504443322'] | ['911', '911', '447322'] | 971504443322 / 91191104452322 | no | Bless 911, Bless 911, I was 0445, 2322. |
| date01 | ['34'] | ['4'] | 430 / 4 | no | And I come on Saturday at the 4th or the evening afternoon. |
| date02 | ['10'] | [] | 10 /  | no | Abu Dhabi, Abu Dhabi, Abu Dhabi, |
| date03 | ['20', '2'] | ['25', '2'] | 252 / 252 | yes | Let us do the 25th of November, or 2pm |

Strict numbers exact 3/17; lenient digit-stream 3/17 (the difference is clips the strict scorer rejects on wording/ordinal/grouping, not on the digits).

## openvino-whisper:whisper-tiny-int8-ov@CPU

**phone01** (scripted) WER 16.692
- ref : my number is zero five zero one two three four five six seven
- hyp : My number is 0, 0, 0, 0, 0, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
- norm ref: my number is zero five zero one two three four five six seven
- norm hyp: my number is zero zero zero zero zero three four zero zero zero zero zero zero zero zero zero zero zero zero zero zero zero zero zero zero zero zero zero zero zero zero zero zero zero zero zero zero zero zero zero zero z
- nums ref ['0501234567'] vs hyp ['00000340000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000'] -> MISS

**phone03** (scripted) WER 16.538
- ref : plus nine seven one five zero four four four three three two two
- hyp : plus 9, 7, 1, and I'm 0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 
- norm ref: plus nine seven one five zero four four four three three two two
- norm hyp: plus nine seven one and i'm zero four four four four four four four four four four four four four four four four four four four four four four four four four four four four four four four four four four four four four fo
- nums ref ['971504443322'] vs hyp ['971', '0444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444'] -> MISS

**spont035** (spont) WER 1.857
- ref : other car rooms are offering cheaper alternatives
- hyp : I will carry on with my own of the new super-ordinary things.
- norm ref: other car rooms are offering cheaper alternatives
- norm hyp: i will carry on with my own of the new super ordinary things

**name03** (scripted) WER 1.4
- ref : it is Abdullah Khalid Rashid
- hyp : And then the Dundas file is done.
- norm ref: it is abdullah khalid rashid
- norm hyp: and then the dundas file is done

**car05** (scripted) WER 1.111
- ref : show me a Honda Accord or a Toyota Camry
- hyp : So, New York, on Dark Orr, or you do Dr. Hengi's.
- norm ref: show me a honda accord or a toyota camry
- norm hyp: so new york on dark orr or you do dr hengi's

**num03** (scripted) WER 1.0
- ref : can you do forty eight thousand
- hyp : Well, in your 240-8000.
- norm ref: can you do forty eight thousand
- norm hyp: well in your two hundred and forty eight thousand
- nums ref ['48000'] vs hyp ['248000'] -> MISS

**num06** (scripted) WER 1.0
- ref : under thirty five thousand and less than sixty thousand kilometres
- hyp : And that 25,000 in the last 10,68,000 kilo meters.
- norm ref: under thirty five thousand and less than sixty thousand kilometres
- norm hyp: and that twenty five thousand in the last ten sixty eight thousand kilo meters
- nums ref ['35000', '60000'] vs hyp ['25000', '78000'] -> MISS

**plate03** (scripted) WER 1.0
- ref : Abu Dhabi twelve eight nine zero one
- hyp : Good.
- norm ref: abu dhabi twelve eight nine zero one
- norm hyp: good
- nums ref ['30'] vs hyp [] -> MISS

**spont002** (spont) WER 1.0
- ref : the weather is really hot today
- hyp : So whether it is a daily heart today.
- norm ref: the weather is really hot today
- norm hyp: so whether it is a daily heart today

**spont027** (spont) WER 0.917
- ref : that is very inconvenient timing for me please show me something else
- hyp : My name is Vey and I am from me. Please show me some of the sounds.
- norm ref: that is very inconvenient timing for me please show me something else
- norm hyp: my name is vey and i am from me please show me some of the sounds

### openvino-whisper:whisper-tiny-int8-ov@CPU: all 12 numeric misses (scripted)

| id | ref numbers | hyp numbers | lenient digit-stream ref / hyp | lenient match | hyp text |
|---|---|---|---|---|---|
| num02 | ['87500'] | ['8000500'] | 87500 / 8000500 | no | The guard has 8,000,500 kilometers. |
| num03 | ['48000'] | ['248000'] | 48000 / 2408000 | no | Well, in your 240-8000. |
| num06 | ['35000', '60000'] | ['25000', '78000'] | 3500060000 / 250001068000 | no | And that 25,000 in the last 10,68,000 kilo meters. |
| num07 | ['72500'] | ['72000', '101'] | 72500 / 720001001 | no | My offer is 72,000 for 100.1. |
| num08 | ['2019', '40000'] | ['2019', '44000'] | 201940000 / 201944000 | no | With 2019 module, 44,000 kilometers. |
| plate01 | ['12345'] | ['8840'] | 12345 / 8840 | no | The crate is the by 8,840 size. |
| plate03 | ['30'] | [] | 128901 /  | no | Good. |
| phone01 | ['0501234567'] | ['00000340000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000'] | 0501234567 / 00000340000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000 | no | My number is 0, 0, 0, 0, 0, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  |
| phone02 | ['0559876543'] | ['1233333'] | 0559876543 / 1233333 | no | Follow me, one two three three three three three |
| phone03 | ['971504443322'] | ['971', '0444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444'] | 971504443322 / 9710444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444 | no | plus 9, 7, 1, and I'm 0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,  |
| date01 | ['34'] | ['4'] | 430 / 4 | no | And I come on Saturday at the 4th or the evening after me. |
| date03 | ['20', '2'] | ['25', '2'] | 252 / 252 | yes | Let us do the 25th of November, our 2PM. |

Strict numbers exact 5/17; lenient digit-stream 5/17 (the difference is clips the strict scorer rejects on wording/ordinal/grouping, not on the digits).

## openvino-whisper:whisper-base-int8-ov@CPU+prompt

**phone02** (scripted) WER 10.231
- ref : call me on zero five five nine eight seven six five four three
- hyp : Bony, Zero, Nissan, Nissan, Nissan, Ford, Ford, Ford, Ford, Ford, Ford, Ford, Ford, Ford, Ford, Ford, Ford, Ford, Ford, Ford, Ford, Ford, Ford, Ford, Ford, Ford, Ford, Ford, Ford, Ford, Ford, Ford, Ford, Ford, Ford, Ford
- norm ref: call me on zero five five nine eight seven six five four three
- norm hyp: bony zero nissan nissan nissan ford ford ford ford ford ford ford ford ford ford ford ford ford ford ford ford ford ford ford ford ford ford ford ford ford ford ford ford ford ford ford ford ford ford ford ford ford ford
- nums ref ['0559876543'] vs hyp ['0'] -> MISS

**car04** (scripted) WER 1.571
- ref : is the Range Rover Sport still available
- hyp : Is there a change in almost what's in your van if you...
- norm ref: is the range rover sport still available
- norm hyp: is there a change in almost what's in your van if you

**spont035** (spont) WER 1.571
- ref : other car rooms are offering cheaper alternatives
- hyp : A book I've always loved, feeling super oil-done with this.
- norm ref: other car rooms are offering cheaper alternatives
- norm hyp: a book i've always loved feeling super oil done with this

**spont038** (spont) WER 1.286
- ref : i want something else that's less showy
- hyp : Al-Han, Someday, I will start the next show.
- norm ref: i want something else that's less showy
- norm hyp: al han someday i will start the next show

**phone03** (scripted) WER 1.154
- ref : plus nine seven one five zero four four four three three two two
- hyp : Plus 911, Plus 971, R044, R322
- norm ref: plus nine seven one five zero four four four three three two two
- norm hyp: plus nine hundred and eleven plus nine hundred and seventy one r forty four r three hundred and twenty two
- nums ref ['971504443322'] vs hyp ['911', '971', '44', '322'] -> MISS

**name03** (scripted) WER 1.0
- ref : it is Abdullah Khalid Rashid
- hyp : Al-Dhawjah,
- norm ref: it is abdullah khalid rashid
- norm hyp: al dhawjah

**place03** (scripted) WER 0.875
- ref : I live in Jumeirah and work in Sharjah
- hyp : Are you having some idea you're working, Sharjah?
- norm ref: i live in jumeirah and work in sharjah
- norm hyp: are you having some idea you're working sharjah

**car05** (scripted) WER 0.778
- ref : show me a Honda Accord or a Toyota Camry
- hyp : Shone on Honda, Accord, or you do that, Carindez.
- norm ref: show me a honda accord or a toyota camry
- norm hyp: shone on honda accord or you do that carindez

**plate02** (scripted) WER 0.75
- ref : plate number Sharjah three four five six seven
- hyp : Big number, Sharjah, T4567
- norm ref: plate number sharjah three four five six seven
- norm hyp: big number sharjah t four thousand five hundred and sixty seven
- nums ref ['34567'] vs hyp ['4567'] -> MISS

**spont029** (spont) WER 0.75
- ref : no no no i don't want this one
- hyp : No, don't apologize.
- norm ref: no no no i don't want this one
- norm hyp: no don't apologize

### openvino-whisper:whisper-base-int8-ov@CPU+prompt: all 10 numeric misses (scripted)

| id | ref numbers | hyp numbers | lenient digit-stream ref / hyp | lenient match | hyp text |
|---|---|---|---|---|---|
| num05 | ['162900'] | ['163900'] | 162900 / 163900 | no | The base is 163,900 |
| num07 | ['72500'] | ['183000'] | 72500 / 72000111000 | no | My off-road is 72,000, 111,000, |
| plate01 | ['12345'] | ['12348'] | 12345 / 12348 | no | The plate is the Bay Area 12,348 |
| plate02 | ['34567'] | ['4567'] | 34567 / 4567 | no | Big number, Sharjah, T4567 |
| plate03 | ['30'] | ['20901'] | 128901 / 128901 | yes | Abu Dhabi, 12, 8901, |
| phone01 | ['0501234567'] | ['50', '4367'] | 0501234567 / 0504367 | no | My number is 050, my number is 4367. |
| phone02 | ['0559876543'] | ['0'] | 0559876543 / 0 | no | Bony, Zero, Nissan, Nissan, Nissan, Ford, Ford, Ford, Ford, Ford, Ford, Ford, Ford, Ford, Ford, Ford |
| phone03 | ['971504443322'] | ['911', '971', '44', '322'] | 971504443322 / 911971044322 | no | Plus 911, Plus 971, R044, R322 |
| date01 | ['34'] | [] | 430 /  | no | And I come on Saturday at the slow-talking the afternoon. |
| date03 | ['20', '2'] | ['25', '2'] | 252 / 252 | yes | Let us do the 25th of November at 2PM. |

Strict numbers exact 7/17; lenient digit-stream 9/17 (the difference is clips the strict scorer rejects on wording/ordinal/grouping, not on the digits).

## faster-whisper:tiny.en-int8+hotwords

**spont028** (spont) WER 3.333
- ref : what is this
- hyp : I appreciate you knowing who you are and what's in this.
- norm ref: what is this
- norm hyp: i appreciate you knowing who you are and what's in this

**name03** (scripted) WER 1.8
- ref : it is Abdullah Khalid Rashid
- hyp : Hey guys, I'm doing a fine day, I'm sure.
- norm ref: it is abdullah khalid rashid
- norm hyp: hey guys i'm doing a fine day i'm sure

**spont036** (spont) WER 1.5
- ref : this sucks
- hyp : Sunk, Sunk, Sunk,
- norm ref: this sucks
- norm hyp: sunk sunk sunk

**spont035** (spont) WER 1.286
- ref : other car rooms are offering cheaper alternatives
- hyp : Boca, rings, and a feeling super-eyed on it.
- norm ref: other car rooms are offering cheaper alternatives
- norm hyp: boca rings and a feeling super eyed on it

**plate03** (scripted) WER 1.143
- ref : Abu Dhabi twelve eight nine zero one
- hyp : Abu Dhabi, 1289 is rolling.
- norm ref: abu dhabi twelve eight nine zero one
- norm hyp: abu dhabi one thousand two hundred and eighty nine is rolling
- nums ref ['30'] vs hyp ['1289'] -> MISS

**plate02** (scripted) WER 1.125
- ref : plate number Sharjah three four five six seven
- hyp : B.A.B.Numbar, Sharjah, T-4567
- norm ref: plate number sharjah three four five six seven
- norm hyp: b a b numbar sharjah t four thousand five hundred and sixty seven
- nums ref ['34567'] vs hyp ['4567'] -> MISS

**car05** (scripted) WER 1.0
- ref : show me a Honda Accord or a Toyota Camry
- hyp : Swami, Arman, Dao, Koh, Riddo, Dao, Dr. Riddi.
- norm ref: show me a honda accord or a toyota camry
- norm hyp: swami arman dao koh riddo dao dr riddi

**name02** (scripted) WER 1.0
- ref : this is Fatima Al Hashimi calling
- hyp : This is the front and has seen the calling.
- norm ref: this is fatima al hashimi calling
- norm hyp: this is the front and has seen the calling

**place03** (scripted) WER 1.0
- ref : I live in Jumeirah and work in Sharjah
- hyp : Al Yevinger, Madan, Working, Shari, Shari, Shari,
- norm ref: i live in jumeirah and work in sharjah
- norm hyp: al yevinger madan working shari shari shari

**spont021** (spont) WER 1.0
- ref : is there something in black
- hyp : Sorento, Sorento,
- norm ref: is there something in black
- norm hyp: sorento sorento

### faster-whisper:tiny.en-int8+hotwords: all 11 numeric misses (scripted)

| id | ref numbers | hyp numbers | lenient digit-stream ref / hyp | lenient match | hyp text |
|---|---|---|---|---|---|
| num05 | ['162900'] | ['16900'] | 162900 / 160900 | no | The base is 160, 900 |
| num06 | ['35000', '60000'] | ['5000', '70000'] | 3500060000 / 50001060000 | no | and Delta 5,000 in the S 10, 60,000 Gmajos. |
| num07 | ['72500'] | ['72000'] | 72500 / 72000 | no | Mayo Ferre, 72,000 flying injured dead ones |
| plate01 | ['12345'] | ['20345'] | 12345 / 812300405 | no | The crate is divided by 8, 12,300, 40-5 |
| plate02 | ['34567'] | ['4567'] | 34567 / 4567 | no | B.A.B.Numbar, Sharjah, T-4567 |
| plate03 | ['30'] | ['1289'] | 128901 / 1289 | no | Abu Dhabi, 1289 is rolling. |
| phone01 | ['0501234567'] | ['3437'] | 0501234567 / 00003437 | no | My number is 0.00 and 0.3437 |
| phone02 | ['0559876543'] | ['1237764'] | 0559876543 / 1237764 | no | Bowing me one, two, three, seven, seven, six, four, what would you? |
| phone03 | ['971504443322'] | ['10', '97150443322'] | 971504443322 / 9197150443322 | no | Last 9, 1, those 9, 7, 1, 5, 0, 4, 4, 3, 3, 2, 2 |
| date01 | ['34'] | ['12'] | 430 / 12 | no | and I come on Saturday at the floor 12 of the afternoon. |
| date03 | ['20', '2'] | ['25'] | 252 / 25 | no | Let us do the 25th of November after PM |

Strict numbers exact 6/17; lenient digit-stream 6/17 (the difference is clips the strict scorer rejects on wording/ordinal/grouping, not on the digits).

## faster-whisper:tiny.en-int8

**spont028** (spont) WER 3.0
- ref : what is this
- hyp : I appreciate what we're doing here. And once it's done.
- norm ref: what is this
- norm hyp: i appreciate what we're doing here and once it's done

**car04** (scripted) WER 1.286
- ref : is the Range Rover Sport still available
- hyp : Is there a change in the support statement of anything?
- norm ref: is the range rover sport still available
- norm hyp: is there a change in the support statement of anything

**spont035** (spont) WER 1.286
- ref : other car rooms are offering cheaper alternatives
- hyp : I work out rooms and are feeling super light on it.
- norm ref: other car rooms are offering cheaper alternatives
- norm hyp: i work out rooms and are feeling super light on it

**car05** (scripted) WER 1.222
- ref : show me a Honda Accord or a Toyota Camry
- hyp : So, in the amount of color do you do, Dr. Inji?
- norm ref: show me a honda accord or a toyota camry
- norm hyp: so in the amount of color do you do dr inji

**car06** (scripted) WER 1.143
- ref : I am interested in a Mitsubishi Pajero
- hyp : I'm interested in a mix of the sheep or shadow.
- norm ref: i am interested in a mitsubishi pajero
- norm hyp: i'm interested in a mix of the sheep or shadow

**car01** (scripted) WER 1.125
- ref : I am looking for a Toyota Land Cruiser
- hyp : I'm looking for the other hand, the case of it.
- norm ref: i am looking for a toyota land cruiser
- norm hyp: i'm looking for the other hand the case of it

**plate01** (scripted) WER 1.0
- ref : the plate is Dubai A twelve thousand three hundred forty five
- hyp : The create is the by 8 121340 size.
- norm ref: the plate is dubai a twelve thousand three hundred forty five
- norm hyp: the create is the by eight one hundred and twenty one thousand three hundred and forty size
- nums ref ['12345'] vs hyp ['921340'] -> MISS

**date02** (scripted) WER 1.0
- ref : book me for tomorrow at ten in the morning
- hyp : Thank you very much.
- norm ref: book me for tomorrow at ten in the morning
- norm hyp: thank you very much
- nums ref ['10'] vs hyp [] -> MISS

**num06** (scripted) WER 0.9
- ref : under thirty five thousand and less than sixty thousand kilometres
- hyp : And at the 25,000 in the next 10,60,000 kilometers.
- norm ref: under thirty five thousand and less than sixty thousand kilometres
- norm hyp: and at the twenty five thousand in the next ten sixty thousand kilometers
- nums ref ['35000', '60000'] vs hyp ['25000', '70000'] -> MISS

**place02** (scripted) WER 0.889
- ref : which branch is closer to Al Quoz or Deira
- hyp : which branch will scroll circle I will close our data
- norm ref: which branch is closer to al quoz or deira
- norm hyp: which branch will scroll circle i will close our data

### faster-whisper:tiny.en-int8: all 11 numeric misses (scripted)

| id | ref numbers | hyp numbers | lenient digit-stream ref / hyp | lenient match | hyp text |
|---|---|---|---|---|---|
| num05 | ['162900'] | ['16900'] | 162900 / 160900 | no | The base is 160, 900. |
| num06 | ['35000', '60000'] | ['25000', '70000'] | 3500060000 / 250001060000 | no | And at the 25,000 in the next 10,60,000 kilometers. |
| num07 | ['72500'] | ['72000'] | 72500 / 72000 | no | My offer is 72,000 flying injured ones. |
| plate01 | ['12345'] | ['921340'] | 12345 / 8121340 | no | The create is the by 8 121340 size. |
| plate02 | ['34567'] | ['4567'] | 34567 / 4567 | no | Big number charger T4567 |
| phone01 | ['0501234567'] | ['1', '567'] | 0501234567 / 001567 | no | My number is 0.01 rotates for 567. |
| phone02 | ['0559876543'] | ['0', '9764'] | 0559876543 / 09764 | no | By me, I'm 0, so I'm 9, 7, 6, 4, that's him. |
| phone03 | ['971504443322'] | ['10', '9715044322'] | 971504443322 / 919715044322 | no | This 9, 1, this 9, 7, 1, 5, 0, 4, 4, 3, 2, 2. |
| date01 | ['34'] | ['12'] | 430 / 12 | no | And I come and have today at the floor 12 of the afternoon. |
| date02 | ['10'] | [] | 10 /  | no | Thank you very much. |
| date03 | ['20', '2'] | ['25'] | 252 / 25 | no | Let us do the 25th of November after PM. |

Strict numbers exact 6/17; lenient digit-stream 6/17 (the difference is clips the strict scorer rejects on wording/ordinal/grouping, not on the digits).

## faster-whisper:base.en-int8+hotwords

**spont028** (spont) WER 6.667
- ref : what is this
- hyp : Ford, Nissan, Man, Ford, Dose, Toyota, Nissan, Ford, Ford, Ford, Honda, Ford, Mustang, Ford, Ford, Ford, Ford, Ford, Ford, Ford,
- norm ref: what is this
- norm hyp: ford nissan man ford dose toyota nissan ford ford ford honda ford mustang ford ford ford ford ford ford ford

**spont035** (spont) WER 1.571
- ref : other car rooms are offering cheaper alternatives
- hyp : and Bercail Williams, and a lot of things you've already done.
- norm ref: other car rooms are offering cheaper alternatives
- norm hyp: and bercail williams and a lot of things you've already done

**plate03** (scripted) WER 1.286
- ref : Abu Dhabi twelve eight nine zero one
- hyp : would have each 12s eight, nine, zero, nine, three, four, four, four,
- norm ref: abu dhabi twelve eight nine zero one
- norm hyp: would have each twelve s eight nine zero nine three four four four
- nums ref ['30'] vs hyp ['12', '89093444'] -> MISS

**phone03** (scripted) WER 1.231
- ref : plus nine seven one five zero four four four three three two two
- hyp : Plus 911, Plus 971, 12044, 3222
- norm ref: plus nine seven one five zero four four four three three two two
- norm hyp: plus nine hundred and eleven plus nine hundred and seventy one twelve thousand forty four three thousand two hundred and twenty two
- nums ref ['971504443322'] vs hyp ['911', '1030222'] -> MISS

**car05** (scripted) WER 1.222
- ref : show me a Honda Accord or a Toyota Camry
- hyp : Show me on my dog or do you do that car in the east.
- norm ref: show me a honda accord or a toyota camry
- norm hyp: show me on my dog or do you do that car in the east

**car04** (scripted) WER 1.143
- ref : is the Range Rover Sport still available
- hyp : Is there a change in over sports in November?
- norm ref: is the range rover sport still available
- norm hyp: is there a change in over sports in november

**name03** (scripted) WER 1.0
- ref : it is Abdullah Khalid Rashid
- hyp : It is a dungna pilot, I'm sorry.
- norm ref: it is abdullah khalid rashid
- norm hyp: it is a dungna pilot i'm sorry

**spont029** (spont) WER 0.875
- ref : no no no i don't want this one
- hyp : If you don't know, don't mind the side.
- norm ref: no no no i don't want this one
- norm hyp: if you don't know don't mind the side

**spont021** (spont) WER 0.8
- ref : is there something in black
- hyp : This is something in the heart.
- norm ref: is there something in black
- norm hyp: this is something in the heart

**num08** (scripted) WER 0.75
- ref : a twenty nineteen model with forty thousand kilometres
- hyp : with 2019, Nollywood, 40,000kms.
- norm ref: a twenty nineteen model with forty thousand kilometres
- norm hyp: with two thousand nineteen nollywood forty thousand kms
- nums ref ['2019', '40000'] vs hyp ['2019', '40000'] -> OK

### faster-whisper:base.en-int8+hotwords: all 8 numeric misses (scripted)

| id | ref numbers | hyp numbers | lenient digit-stream ref / hyp | lenient match | hyp text |
|---|---|---|---|---|---|
| plate01 | ['12345'] | ['81345'] | 12345 / 81345 | no | The crate is by 81345. |
| plate02 | ['34567'] | ['4567'] | 34567 / 4567 | no | Beat number, Sharjah, T4567 |
| plate03 | ['30'] | ['12', '89093444'] | 128901 / 1289093444 | no | would have each 12s eight, nine, zero, nine, three, four, four, four, |
| phone01 | ['0501234567'] | ['76'] | 0501234567 / 0501034567 | no | My number is 050 1034567 |
| phone02 | ['0559876543'] | ['5', '128764', '14'] | 0559876543 / 512876414 | no | 5 new, 1, 2, 8, 7, 6, 4, 14 |
| phone03 | ['971504443322'] | ['911', '1030222'] | 971504443322 / 911971120443222 | no | Plus 911, Plus 971, 12044, 3222 |
| date01 | ['34'] | ['4'] | 430 / 4 | no | and I come on Saturday at the 4th working the afternoon. |
| date03 | ['20', '2'] | ['25', '2'] | 252 / 252 | yes | That is 25th of November at 2 PM. |

Strict numbers exact 9/17; lenient digit-stream 10/17 (the difference is clips the strict scorer rejects on wording/ordinal/grouping, not on the digits).

## moonshine:base

**spont028** (spont) WER 4.333
- ref : what is this
- hyp : But I'm not appreciating the people who are in the village and myself. Is this...
- norm ref: what is this
- norm hyp: but i'm not appreciating the people who are in the village and myself is this

**spont035** (spont) WER 1.857
- ref : other car rooms are offering cheaper alternatives
- hyp : A book I'll read is a love-threatening tutor I'll tell you this.
- norm ref: other car rooms are offering cheaper alternatives
- norm hyp: a book i'll read is a love threatening tutor i'll tell you this

**name03** (scripted) WER 1.4
- ref : it is Abdullah Khalid Rashid
- hyp : It is a do not buy a damn shirt.
- norm ref: it is abdullah khalid rashid
- norm hyp: it is a do not buy a damn shirt

**phone02** (scripted) WER 1.308
- ref : call me on zero five five nine eight seven six five four three
- hyp : By the name of the author, I am now known to have been successful before that year.
- norm ref: call me on zero five five nine eight seven six five four three
- norm hyp: by the name of the author i am now known to have been successful before that year
- nums ref ['0559876543'] vs hyp [] -> MISS

**num03** (scripted) WER 1.0
- ref : can you do forty eight thousand
- hyp : Woke in U-248,000.
- norm ref: can you do forty eight thousand
- norm hyp: woke in u two hundred and forty eight thousand
- nums ref ['48000'] vs hyp ['248000'] -> MISS

**spont001** (spont) WER 1.0
- ref : i didn't say that can you read my question properly
- hyp : 
- norm ref: i didn't say that can you read my question properly
- norm hyp: 

**spont010** (spont) WER 1.0
- ref : this is quite ugly
- hyp : 
- norm ref: this is quite ugly
- norm hyp: 

**spont021** (spont) WER 1.0
- ref : is there something in black
- hyp : 
- norm ref: is there something in black
- norm hyp: 

**spont029** (spont) WER 1.0
- ref : no no no i don't want this one
- hyp : 
- norm ref: no no no i don't want this one
- norm hyp: 

**spont036** (spont) WER 1.0
- ref : this sucks
- hyp : 
- norm ref: this sucks
- norm hyp: 

### moonshine:base: all 11 numeric misses (scripted)

| id | ref numbers | hyp numbers | lenient digit-stream ref / hyp | lenient match | hyp text |
|---|---|---|---|---|---|
| num02 | ['87500'] | ['8', '7500'] | 87500 / 87500 | yes | The car has eight or seven thousand five hundred kilometers. |
| num03 | ['48000'] | ['248000'] | 48000 / 248000 | no | Woke in U-248,000. |
| num05 | ['162900'] | ['1163100'] | 162900 / 1630001000100 | no | The banks had 163,000 million hundred. |
| num06 | ['35000', '60000'] | ['60000'] | 3500060000 / 60000 | no | Under the fact that thousands in less than sixty thousand kilometers. |
| plate01 | ['12345'] | ['12340'] | 12345 / 12340 | no | The crate is Dubai Aid, twelve thousand, three hundred forty-size. |
| plate02 | ['34567'] | ['567'] | 34567 / 567 | no | Blade number Sharjah T. Seoul, five, six, seven. |
| plate03 | ['30'] | ['8'] | 128901 / 8 | no | Udhabi tells eight nines in Iran. |
| phone01 | ['0501234567'] | ['0501034367'] | 0501234567 / 0501034367 | no | My number is 050-103-4-367. |
| phone02 | ['0559876543'] | [] | 0559876543 /  | no | By the name of the author, I am now known to have been successful before that year. |
| phone03 | ['971504443322'] | ['911', '971504403322'] | 971504443322 / 911971504403322 | no | Last nine, one, one, last nine, seven, one, five, zero, four, four, zero, three, three, two, two. |
| date01 | ['34'] | [] | 430 / 4 | no | And I come on Saturday at the fourth or the end of the afternoon. |

Strict numbers exact 6/17; lenient digit-stream 7/17 (the difference is clips the strict scorer rejects on wording/ordinal/grouping, not on the digits).
