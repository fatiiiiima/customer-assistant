# The voice agent: everything so far (transcript of the video)


## Welcome

**[0:00] The voice agent: everything so far**

Hello. This is a spoken walkthrough of everything we have built, measured and decided on the voice agent in this session. It runs about forty minutes, in chapters, so you can stop and come back. I will say plainly what has been measured, what is only designed, and what has not been tested yet. That matters, because a few of the numbers come from a laptop and a few come from a simulation, and they should not be mixed up.

**[0:23] The chapters**

Here is the plan. First, what the system is and what we were trying to achieve. Then latency, meaning where the time goes in one turn. Then how many callers the laptop can carry, and why it runs out. Then the tests on the Mac's graphics chip, and the work on one thousand callers. After that, the move to Redis and the engine acting as a model, the advise tool, a long chapter on autoscaling step by step, a close look at the engine itself, and finally why each technology was chosen and where everything stands.


## The system

**[0:54] What the system is**

The system is a voice agent for a used car dealership. A customer presses the voice button on the website. The page asks a router for a worker. The router gives it the address of one worker, and the page opens a live connection straight to that worker. The worker is a Python program that listens to the microphone audio, works out when the caller has finished speaking, sends the audio to a speech recogniser, and hands the words to the engine. The engine decides what to do, looks things up in the stock database, and has a language model write the reply. A voice server turns the reply into speech, and the audio goes back to the browser.

**[1:31] The goal**

The goal was to get the delay between the caller finishing a sentence and hearing the first sound of the answer below roughly one second, and then to show the system can serve many callers at once. The ask from your boss, in the end, was one thousand to two thousand callers. A rule I have tried to follow throughout is to report what was really measured, to say when something was a guess, and to correct myself when I got something wrong. I will point out the corrections as we go.


## Latency

**[1:58] Where the time goes in one turn**

Let us start with one turn. The clock starts when the caller stops speaking. First, a speech detector waits for zero point two seconds of silence. Then a small turn model, called Smart Turn, takes about four hundredths of a second to decide whether the sentence is really over, so a pause in the middle of a sentence is not taken as the end. Then the recogniser turns the audio into words, which takes about one hundred and seventy to two hundred milliseconds. Next the engine decides what to do. When code rules decide, that takes three thousandths of a second. When the model decides, it takes about two tenths of a second. Then the model writes the reply, and the first words arrive after two to six tenths of a second. Finally the voice makes the first sound in about one tenth of a second. Add them up and you land near one second.

**[2:47] What made it faster**

Several changes cut that time. The first is to start work during the wait. While Smart Turn is still deciding whether the caller has finished, the engine already starts working out the reply for the words it has so far. The second is an early start for the recogniser. It starts reading the audio after only a tenth of a second of silence, and its early reading is verified against the final one. The third is an opener. A short cached word like Sure is played at once while the model is still writing, so the caller hears something right away. The fourth is to let the voice start after only three words of the first clause. Together these brought the median in our laptop tests down to about zero point eight seconds.

**[3:29] What we tried and turned off**

Not everything worked. A text check on the early transcript, and a shorter hold setting, both made the agent cut people off. A caller saying, can you tell me, with a question mark, was split into two questions. We tried it at four tenths and even five tenths of a second and it still split sentences. So both are left off by default, and the notes say they are harmful. It is useful to know what does not work, so nobody tries them again.

**[3:54] Problems found on live calls**

Your own live calls found real problems. When you paused between words, the agent took the parts as separate questions. Barge-in, where you speak over the agent, was not reliable. Sometimes it stopped talking midway. A bug made it read the same three cars again when you asked to keep going, which we fixed by paging through the list. And you asked that it remember the whole conversation until you close the browser, not only fifteen minutes. The call is now kept for a day, and the page shows the real figure.

**[4:26] What the page shows**

The website page now shows a table of timings for every turn, with a breakdown of the reply words step: how long the request took to reach the model, how long the model took to produce its first text, and so on. The page also keeps a call id in the browser tab, so a reload carries on the same conversation, and it shows a message after a quiet spell. About two hundred milliseconds of the reply time is the trip to the model in the United States. The rest is the model's own thinking and writing time.


## Load

**[4:56] How many callers: the first test**

Now, how many callers can it serve? We ran synthetic callers through the real WebSocket door, each with an open microphone, using the live settings. Up to eight callers at once, the median wait stayed between nine tenths and one point one seconds. At twelve a slow tail appeared. At sixteen all together, it broke, with waits of two point four to two point nine seconds. One correction here: I said earlier the Mac has six cores. It has twelve, six performance and six efficiency. The twelve is the right number.

**[5:28] Why it breaks: three limits**

Why does it break? It was a question whether the cause was a bug or the laptop. The answer is the limits of the hardware and the shape of the services, not the call logic. There are three limits. First, the laptop's twelve cores are shared between the workers, the voice server, the recogniser and the test callers. Second, the voice server and the recogniser each handle one request at a time. If two requests arrive together, the second waits. More cores do not fix that, you need more copies or batching. Third, each worker is one Python process with one event loop, which cannot carry too many calls.

**[6:05] What each service can carry**

We measured each shared service on its own. One copy of the voice server makes about seven seconds of speech for every second of work, enough for about thirteen callers. One copy of the Qwen3 recogniser handles about six clips a second, about twenty three callers. The Parakeet recogniser handles about twenty three clips a second, about eighty nine callers. And Bedrock, the cloud model, handled at least sixty requests a second over plain connections.

**[6:32] What one turn costs**

We also counted what one turn asks of the shared services. A turn makes about one and a half model requests. It sends two and a half to nearly three clips to the recogniser, about seven seconds of audio, because of the early start. And the voice makes about five and a half seconds of speech. A worker uses about a fifth of one core per talking call, which matches the deployment plan of three to four cores for sixteen calls. Memory stayed flat, between twenty six and thirty one gigabytes, from one to sixteen callers, so memory was not the limit.

**[7:05] The simulation**

To go beyond what a laptop can run, we built a simulation using the measured speeds. It matches the eight caller run and is a little optimistic at sixteen. For one hundred callers it needs eight voice servers and five Qwen3 recogniser copies, or two Parakeet. For two hundred callers, sixteen voice servers and nine Qwen3, or three Parakeet. For one thousand callers, eighty voice servers and forty five Qwen3, or fifteen Parakeet, with up to seventy two workers. The answer time at the ninetieth percentile stays under one second. With half that hardware, answers collapse to about nine seconds. This is a model, not a real test, and it says so.

**[7:44] Two things hidden at first**

Two problems were hidden at first. At eight callers, the fallback voice was quietly saying some sentences when the main voice was too slow, which made the numbers look better than they were. And when many requests shared one HTTP version two connection, some were dropped under concurrency. In the real worker it was only about one percent at sixteen callers, but it was found by testing each service by itself.


## The Mac GPU

**[8:08] Can the Mac's GPU help?**

You asked whether we could test moving things from the processor to the graphics chip on this Mac, because a GPU is what you will switch to. We tried both. Here is what we found.

**[8:19] The voice on the GPU**

First, the voice. We ran the Pocket text to speech model on the processor and on the Mac's graphics chip. On the processor it made about seven point eight seconds of speech for each second of work. On the graphics chip, four point four. The first sound took thirty five milliseconds on the processor and a hundred and sixty four on the graphics chip. So the graphics chip was slower on every measure. The reason is that the model makes sound in many tiny steps, one sentence at a time, and a graphics chip loses time starting each small step. The package can only generate one sentence at a time, so the thing that makes a GPU worthwhile, many sentences in one step, is not available. So the voice stays on the processor here.

**[9:01] The recogniser on the GPU**

Second, the recogniser. The Qwen3 recogniser already runs on the Mac's graphics chip. We tried decoding several clips together in a batch, and the gain was only one point two to one point three times. Then we tried reusing the first part of the prompt. Every clip starts with the same text, the list of car names. Computing that part once and starting each clip from a copy made recognition one point six times faster, about a hundred and thirty milliseconds a clip instead of nearly two hundred. On sixty eight clips, the text was identical on sixty six. The two that differed were silence and noise. I need to correct myself here: my first benchmark passed the hot words as one long string, so they were split into single letters, and that made the results look much more dramatic. The numbers I give now are with the real list. The recogniser server in the project now keeps the beginning of the prompt, with tests, and it switches itself off safely if anything fails.

**[9:59] Why the worker stays on the processor**

Third, you asked whether the workers could use the GPU. The two small models inside the worker are tiny. The speech detector costs about half a percent of one core per call and the turn model about one percent, against about twenty percent of a core for the whole worker on a talking call. The rest is ordinary Python code moving audio and messages, checking state, sending requests and writing to the database. A GPU runs big blocks of identical arithmetic, and that kind of code is not that. It is not strictly impossible. You could batch the audio maths for hundreds of calls in one GPU process. But you would redesign a lot to save very little. So the workers stay on cheap processor machines, and the GPU goes where the heavy models are.

**[10:45] Why servers handle one request at a time**

You asked why the voice and recogniser servers handle one request at a time. Each holds one copy of the model, and its state is not safe to share between two requests. Running two at once on the same cores just splits the compute, so each takes twice as long. The lock mostly reflects that, it does not cause it. The ways out are batching, where several requests are decoded together, or running more copies. A server GPU running vLLM batches by design, which is why the deployment plan uses it. And when we removed that limit in the thousand caller test by using fast stand ins, the Mac still stopped well short of a thousand talking callers, which shows the worker side limits it too.


## One thousand callers

**[11:25] Your boss's claim**

Your boss said that since this Mac has thirty six gigabytes of memory, it should be able to run a thousand callers, because he has done it before. On memory he is right: we measured about three and a half megabytes per connected call, so a thousand calls fit easily. What runs out is processor time and the number of processes. A graphics chip on a Mac shares the same memory as the processor, so it does not add any.

**[11:51] The 1000 connected calls test**

So we built a test. It runs the real router, four real workers, a real database and the real model, with a fast stand in for the recogniser and the voice so they do not limit the result. One thousand quiet callers connect over a hundred and fifty seconds, and four real conversations run before and after. All one thousand were held open, none refused, none dropped. The router assigned each in about a millisecond. The wait for an answer for the real conversations went from forty eight hundredths of a second alone to fifty two hundredths with one thousand connected. So the latency did not drop, and it did not get worse. It stayed flat, which is what a healthy system should do.

**[12:30] The big caveat**

Now the caveat, and it is important. Those thousand callers were quiet. They sent one short burst of silence a second. A real browser microphone sends fifty a second, and that costs the worker much more. With open microphones, a call cost about two and a half to three percent of a core. At four hundred open microphones across four workers, the wait for real conversations rose from about half a second to seven seconds, even though the machine was only sixty percent busy. The reason is that one worker process cannot hold about a hundred calls at once. Production allows sixteen calls per worker for exactly this reason. So the test shows that the router, the workers' connection handling and the database cope with a thousand connections. It does not show a thousand people talking.

**[13:16] The user journeys script**

Your boss then suggested a script with different task agents, browsing the web page, talking to the agent, booking a test drive, run many times. We built that. Each agent is one simulated person. Browsers open pages with a pause to read each. Talkers call the agent and ask questions. Bookers call and book a test drive, each with a different day, time, name and phone number, because the first version used identical details and the server correctly refused the duplicates. With one thousand people, nine hundred and seventy browsing, twenty talking and ten booking, there were over fifty six thousand page requests and none failed. One hundred and ten calls completed and none was refused or failed. Thirty calls were open at the same time at the peak, and the median wait was forty four hundredths of a second.

**[14:04] The same script with the real services**

We also ran the script with the real recogniser, voice and model, with three hundred and eight people. Nineteen calls completed, and eight of nine booking calls really booked a test drive, with a median wait of about one second. That shows the real parts work together end to end. The thousand person run shows the rest of the system holds when most people only browse, as on a real site. Neither shows a thousand people all talking.

**[14:30] What a thousand talking callers needs**

So what would a thousand talking callers need? About sixty three to seventy two worker processes at sixteen calls each, which is roughly a hundred and eighty processor cores if the lines are mostly quiet, and up to two hundred if everyone talks constantly. About eighty voice server copies, around a hundred and sixty cores. A recogniser that batches on a GPU, or fifteen to forty five of the Mac style copies. And the model on Bedrock handling about a hundred requests a second. A Mac with twelve cores and a single set of services cannot hold that, so it needs rented machines.

**[15:05] The voice and the GPU, again**

Your boss might say, why not move the voice and workers to the GPU. Here is the arithmetic. A talking caller needs about five and a half seconds of speech per turn, and a turn comes about every fifteen seconds. So a thousand callers ask for about three hundred and seventy seconds of speech every second. At seven point eight seconds of speech per second per copy, that is about forty seven copies running flat out. At the graphics chip's four point four, about eighty four. One Mac cannot be that many copies.

**[15:37] Renting machines and what it costs**

You asked how to rent the machines and what it would cost. This is an order of magnitude estimate from list prices I remember, not a quote, so check it with the provider's calculator. Start the quota requests first, because limits on processors, GPUs and model throughput can take days to approve. Run a small cluster from the existing deployment files with the recogniser on a GPU machine, ramp in stages from a hundred to two thousand callers, and shut everything down when the run ends. For a four to six hour session, machines would cost about a hundred to two hundred dollars for a thousand callers and two to four hundred for two thousand. The model's per token charges are the big unknown, because I do not have a reliable price for that model on its fast tier. A realistic budget to ask for is three hundred to a thousand dollars for a thousand caller day. One more catch: the recogniser we measured is Mac only, so a cloud run would also measure the vLLM Whisper setup, which has no measurements yet.

**[16:34] What to tell your boss**

What can you tell your boss? What was proven: the router, workers and database held a thousand connected calls on one Mac, latency was flat up to eight talking callers, and the simulation says a thousand callers stays under a second with about eighty voice servers and a batching recogniser. What was not shown: a thousand people all talking on this laptop. And ask him exactly what he ran before: how many were connected versus talking, on what machine, and whether the recognition and voice ran locally or came from a cloud service. If the heavy parts were remote, a thousand on one machine is plausible, but that is a different architecture from ours.


## Redis

**[17:13] Redis for worker reports and call memory**

Your boss wanted Redis, so the stock database is not also holding what callers said. The code already supported it, so what we added is the way to run and deploy with it. The cluster launcher now starts a local Redis by default, and the deployment file reads its address from its own setting. PostgreSQL now holds only the stock and the bookings. Bookings still carry a caller's name and number, so that part of the privacy concern remains. Each worker's report is a key with a six second expiry, which is how a dead worker drops out of the list by itself. And call memory is saved after every turn with a twenty four hour expiry.

**[17:49] Conversations are remembered**

To remember conversations, the local Redis keeps an append only file on disk, written once a second. We checked it with the real services. A caller asked about the Land Cruiser on one worker and hung up. On the other worker, asking how much was that one again, got the answer, one hundred and eighty nine thousand dirhams. Then Redis was stopped, the launcher started it again, and back on the first worker, asking what colour was that one, got white. For production you would use a managed Redis with a password and encryption. We have not tested a real managed Redis, or a failover.


## Model as the engine

**[18:23] Letting the model decide every turn**

Your boss said the engine should be a model too, because the user can ask whatever they like. Today the model already writes every reply. The code rules choose which lookup to run in about four of five turns. We switched that off so the model chooses every turn, and measured with four callers, two runs each. The median wait went from about seventy eight hundredths of a second to about one point two seven seconds. The ninetieth percentile went from about one and a half to about two seconds. Model requests per turn went up by sixty to seventy percent, and the model input by about sixty percent. So letting the model decide everything costs about half a second, and more model load.

**[19:04] What the model decision gains, and the risk**

What does it gain? We asked six off script questions in both settings. The model deciding setting answered a comparison, which is better for a family of five, and answered Paris for the capital of France, where the other said, I do not have that info. But both settings invented policy. They said we can arrange instalments, and we wash every car before delivery. Neither is in the stock or policy data. The checks that catch invented numbers and names do not catch invented claims in general. A written list of what the dealership does and does not offer, given to the model, would close that.


## The advise tool

**[19:40] The gap: needs, not cars**

You also said the model should go into the database when someone asks for a specific car, and when someone just says they are looking for a car, it should ask what they need or suggest cars for different requirements, as flexible and conversational as possible, within limits. A named car already went to the database through a search tool. The gap was in between. A caller who described a need, something for the family, something sporty, a long road trip with the kids, was sent to a plain search that has nothing to filter on, so the same list came back, or the reply invented praise such as quick and fun.

**[20:14] How advise works**

The fix is a new tool called advise. The model says that this is a need, and which one. Then code chooses up to three cars by recorded figures from the stock and the site's spec sheets: seats and boot space for a family or a trip, horsepower and zero to a hundred time for sporty, hybrid then fuel use for economy, price and year for luxury and budget, one car per model. Each car goes to the model with a reason built from those figures. For an open request like, I am looking for a car, the model is told not to list cars but to ask one question: what they will use it for, how many people, or their budget. The model must not call a car popular, a best seller, great or reliable, because there are no sales figures in the stock. To say best seller, the dealership would have to supply a sales or featured flag.

**[21:04] What happened live**

Live, with the real recogniser and model: I am looking for a car got, what kind of car are you after, and how many people will be riding with you. Something good for a family got the Kia Carnival and the Chevrolet Tahoe, seven seats. Something sporty and fast got the Mercedes G sixty three and the Range Rover Sport. And the cheapest one you have now gets the Kia Rio at thirty nine thousand. Two other gaps were found on the way: the model's tool list was missing the best of tool, so superlatives went to a search limited to the car just discussed, and the stock in the local database was an old twenty four car seed with no spec sheets. It is now the full seventy four cars with their spec sheets. The cost is time. A suggestion turn took about one point four to two point one seconds, against about one second for a turn that names a car.

**[21:54] Still wrong sometimes**

It is not perfect. The model still adds praise the figures do not give, such as, both great for families. When a single car is found, it sometimes reads out the whole spec sheet. It asks, which one fits your budget, too often. And we have not counted how often it picks the right need. Those are known weaknesses, written in the notes.


## Autoscaling

**[22:16] The autoscaling diagram: two loops**

Now a long chapter on the autoscaling diagram, step by step. The diagram is a file in the docs folder you can open in a browser. The idea first. There are two loops. The call loop is fast and happens for every call: a page asks the router for a worker, connects to it, and talks. The scaling loop is slow and runs about every fifteen seconds: it reads how busy the workers are and changes how many workers exist. The two loops connect in just a few places. Workers report their load, the load drives scaling, and new workers appear in the router's list by reporting. The design choices to defend are these. Audio never goes through the router. Each call stays on one worker for its whole length. Call memory lives in Redis so another worker can continue a conversation. And scaling follows demand, including callers who were refused, not processor use.

**[23:10] The boxes and what they are made of**

What are the boxes? The caller's page is JavaScript in the browser, with a WebSocket. The router is our Python code, using FastAPI, two small pods. Redis is the Redis server, and our Python talks to it through a library. The worker pod is our main code, in Python with FastAPI and the Pipecat audio library. Inside it are the speech detector and turn model, small ONNX models, the engine, and the voice. It runs as a container with three to four cores and six to eight gigabytes, from eight to seventy two copies. The call cap and the quality governor are Python classes inside the worker. The model server and Whisper are open source vLLM servers on GPUs. Prometheus is open source monitoring, the metrics adapter is a Kubernetes add on with one of our rules, the autoscaler is built into Kubernetes with our settings, and Kubernetes runs everything.

**[24:03] Step 1: ask for a worker**

Step one. The page asks the router for a worker, once per call. The router answers from a list it already holds in memory, so it replies in about a millisecond. It computes each worker's free room as its cap, minus its active calls, minus calls it handed that worker in the last three seconds. It drops workers that are full or draining, draws two at random and gives the call to the one with more room. This avoids sending a whole burst to the one worker that looked emptiest. If every worker is full it answers busy, and the page offers text chat.

**[24:37] Step 2: the call goes straight to the worker**

Step two. The page connects straight to the address it was given, through the ingress to that one pod. Audio never passes through the router, so the router can never become a bottleneck and can run as several copies. The call stays on that worker for its whole length.

**[24:53] Step 3: each worker reports its load**

Step three. Every two seconds, and about a fifth of a second after a call starts or ends, each worker writes a short report into Redis. It holds the worker's id and address, its calls now, its call cap, a draining flag, and its recent answer time. The report has a six second expiry. A worker that crashes or is stopped stops renewing it, and drops out of the list by itself within six seconds, with nobody having to notice.

**[25:20] Step 4: the router reads the list**

Step four. Twice a second the router copies the whole worker list from Redis into memory. That is why an assignment takes about a millisecond even with a thousand callers arriving. It never asks Redis during a call. If Redis fails, the router keeps its last good list.

**[25:37] Step 5: each worker sets its own cap**

Step five. Each worker sets its own cap, the number of calls it will take, with a ceiling of sixteen. Every five seconds it looks at the ninetieth percentile of its answer times over the last thirty seconds, if it has at least twelve samples. Slower than one second, and the cap drops to seventy five percent. Faster than nine tenths of a second while the cap is what limits it, and the cap rises by one. Calls already running are never dropped. And if the replies after a cut are not clearly faster, the cut is undone and no cuts happen for a minute, because the slowness is not from this worker's load, and turning callers away would cost them the call for nothing.

**[26:16] Step 6: shared services and the governor**

Step six. Recognition and the model run on shared GPU servers. The quality governor inside each worker cannot see those servers. It watches how long the model takes to answer that worker's own requests, measured with a clock in plain Python around each model step. If the first sentence takes longer than seven tenths of a second, turns step down: code says the facts at once and the model adds only a short follow up. If decisions take longer than forty five hundredths of a second, a growing share of turns, up to ninety percent, skip the model and are answered by code. A small test request every twenty seconds shows when the model has recovered. An alert fires when more than twenty percent of turns over five minutes were answered by code. A caveat: a slow network looks the same as a slow model, and each worker judges alone.

**[27:06] Step 7: Prometheus collects the numbers**

Step seven. Prometheus collects the numbers, and the important point is that nothing is sent to it. It pulls. Every fifteen seconds it makes a plain web request to each worker's metrics page and the router's metrics page, and stores what comes back with a timestamp. The workers and the router do not know it exists. Each worker's page lists calls in progress, its call cap, answer times, and counts of quality levels. The router's page includes the number of callers it turned away in the last thirty seconds. That one matters most: a refused caller never reached a worker, so no worker knows about them. In Kubernetes the pods carry annotations so Prometheus can find them. One caveat: the Prometheus file in the repo is the docker compose version, and the Kubernetes scrape setup is not in the repo.

**[27:55] Step 8: the demand ratio**

Step eight. The metrics adapter turns all this into one number, the demand ratio. It is the calls in progress, plus the callers turned away in the last thirty seconds, divided by the sum of all the workers' caps. With ten workers and a total cap of one hundred sixty, eighty calls gives point five. A hundred and twelve calls gives point seven, the target. A hundred and sixty calls gives exactly one, every slot used and nobody refused yet. A hundred and sixty calls with eighty refused gives one point five. And caps in the bottom matter: if the workers cut their caps to eight each because replies got slow, the same eighty calls now give a ratio of one. A correction to the page's wording: exactly one means full, and only above one are callers being refused.

**[28:42] Why count the turned-away callers**

Why count the refused callers at all? Without them the ratio could never go above one, because calls in progress cannot exceed the caps. The autoscaler would see one against a target of point seven and ask for about forty three percent more pods, whether five callers or five hundred had been refused. With the refused callers on top, the ratio can climb to two or three, and the autoscaler asks for proportionally more pods in one step. And because the count covers only the last thirty seconds, it rises as soon as refusals start and drops within half a minute of enough capacity arriving. One limit: a caller who is refused and tries again counts twice, which exaggerates demand a little.

**[29:22] Step 9: the autoscaler compares with 0.7**

Step nine. The autoscaler compares the ratio with a target of point seven, every fifteen seconds. Above it, it asks for more pods, in the simple form new pods equal current pods times the ratio over point seven, at most doubling every fifteen seconds, never above seventy two. Below it, it waits five minutes of low demand, then removes at most two pods a minute, never below eight. Calls last minutes, and cutting one off is worse than paying for an idle pod. Kubernetes has some extra details in the real controller, such as a default tolerance of about ten percent where it does nothing, and I have not modelled that in the page.

**[30:02] Why the target is 0.7**

Why point seven? The deployment file says the autoscaler keeps workers about seventy percent full, and sizes the pod limit as eight hundred callers over sixteen times seventy percent. The reasoning I infer is this. A new pod needs a minute or two to load its speech models, and meanwhile callers keep arriving at the pods that exist, so there must be spare room. Workers also get slower as they fill. The cost is paying for about thirty percent more pods. For five hundred callers that is about forty five pods instead of thirty two. I could not find a measurement that picked point seven over another value, so it is a design choice, not a tuned number. A real cluster test with real pod start times would justify changing it.

**[30:44] Step 10: Kubernetes starts pods**

Step ten. The autoscaler gives Kubernetes a new pod count, and Kubernetes starts the pods. A new pod pulls the image, mounts the speech models and loads them, which takes a minute or two. It is not ready until its health check answers, and meanwhile callers keep arriving at the pods already running. That delay is exactly why the target leaves headroom.

**[31:07] Step 11: a new pod joins**

Step eleven. From its first report to Redis, the new pod is in the list. Within half a second the router can give it calls, and the two at random draw sends it a share at once because it is the emptiest. Nobody has to register it.

**[31:21] Step 12: scaling down**

Step twelve. When Kubernetes stops a pod, the worker marks itself draining and tells Redis at once, not at the next beat. The router stops giving it new calls, but calls in progress go on. Kubernetes waits up to fifteen minutes for them to finish before the pod is removed, so scaling down never cuts a caller off. That is also why scale down is slow. One nuance: a draining worker reports a cap of zero, but its remaining calls still count in the ratio, so the ratio is slightly higher while pods drain.

**[31:52] A burst, minute by minute**

The page also has a simulation of a burst, using the real rules. Demand rises from a hundred to seven hundred callers in two minutes. Nine pods hold a hundred and forty four calls, so the ratio jumps to about two point eight. The autoscaler doubles every fifteen seconds and within forty five seconds has asked for seventy two pods, but the new pods take ninety seconds to load, so about two hundred and fifty callers are turned away for roughly ninety seconds. Notice the overshoot: the ratio is measured against only the ready pods, so it asks for more pods than the burst finally needs, which is deliberate over provisioning, and then scale down is slow, about twenty eight minutes from seventy two pods down to fourteen. The ninety second start time is my assumption, and the whole simulation is an illustration of the rules, not a measurement of a real cluster.

**[32:42] What happens when something fails**

What happens when something fails? If a worker crashes mid call, the call drops, and its report expires within six seconds. The snapshot in Redis lets the caller continue elsewhere. I tested the hang up and return case, not a crash. If Redis is down, calls in progress carry on because audio never uses Redis. Reports and snapshots are lost, and the router keeps its last list. If the router is down, new calls cannot be assigned, calls in progress are unaffected, and there are two copies. If the model is slow, the governor lowers quality, the cap shrinks, the ratio rises and pods are added. If Prometheus or the adapter is down, the autoscaler has no metric and holds the current pod count. And if the pod limit of seventy two is reached, callers are turned away and an alert fires after two minutes.

**[33:34] Gaps to know about**

Some gaps to know about. The autoscaler only scales workers. The model server and Whisper are fixed at two replicas each, sized for five hundred callers, and have no autoscaling. At the point seven target, seventy two pods covers about eight hundred callers, so for a thousand you need about ninety, and the limit must be raised. The simulation ignores the autoscaler's default tolerance and treats every pod as holding sixteen calls. The deployment file starts workers with a prefix flag, which its help text calls experimental, and I have not checked how that interacts with the governor's automatic levels. And I have not checked what the page does if the assigned worker cannot be reached. The deployment files have never been run on a real cluster.


## The engine

**[34:19] What the engine is**

Now the engine, the box inside the worker. It is our own Python class, called AgentEngine, about twenty six hundred lines, plus helper modules. It is not a single model. It is the part of the worker that, for each thing the caller says, decides what to do, looks things up in the stock and bookings, and produces the words to speak. A model, Nemotron on Bedrock in development and Qwen3 fourteen billion on vLLM in the plan, does two jobs inside it: choosing an action, and writing the reply words. Everything else is code. Each call has a Session object that holds the call's state in memory, and it is saved to Redis after every turn.

**[34:58] Who does what**

The division of labour is the key. The model chooses which action to take, but in about four of five turns the code's own rules choose. The code, and only the code, reads the details out of the caller's words: filters, car ids, dates and times with a date parser, phone numbers, and names. Code looks up the stock, the free slots and the bookings. Code decides what is true, as the facts for the turn. The model writes the spoken sentences in its own words, and code checks each sentence. And only code makes a booking, and only after a read back and a yes.

**[35:34] One turn through the engine**

Here is one turn. The transcript arrives. Outside the test mode, the engine first catches junk, like a fragment, or an unfinished phrase. Then it picks an action, by rules or by a short model call that returns just a tool name. Code fills in the arguments. Code runs the action against the service layer, which searches the stock with a plain filtered query. The result has the standard wording, a list of true fact lines and the question the system is waiting for. Then the quality level is chosen, an opener may play, and the model writes the reply from the conversation and the facts, sentence by sentence. Finally the exchange is saved and the reply is spoken.

**[36:14] The actions the model can choose**

The model can choose from about twenty actions. Search cars, advise, get car, best of and compare cars are about the stock. Check slots, prepare booking, confirm booking and cancel draft are the booking steps. My bookings, cancel booking and reschedule booking are about existing bookings. Showroom info gives hours and address. And repeat, hand off to a person, end the call and none cover the rest. The model sees the time, the car in focus, the last cars listed and the last few lines of the call.

**[36:47] How a sentence is checked**

Each sentence the model writes is checked before it is spoken. Plain characters only and not too long. No number that is not in the facts, the caller's words or the earlier conversation. No capitalised name that is not there. No wrong seat count. No colour the facts do not give. No claim or commercial word such as warranty, discount or finance that the facts do not say, and no statement that a visit is booked when the system has not said so. A second, stricter check ignores the history. If a sentence fails, the writer is asked again once. If that fails too, the code's own wording is spoken. The model never sees the floor price, and the car objects it is given do not contain one.

**[37:30] Booking safety**

Booking is a guarded sequence. The prepare booking action collects the day and time, the name and the phone. The engine reads them back. Only a yes triggers the confirm booking action, and even then the service re checks that the slot is free and the caller has no duplicate booking. If the slot was just taken it offers the nearest alternatives. A reschedule cancels the old booking only as part of the move, and puts it back if the new slot fails.

**[37:57] Test mode against production mode**

A very important point. The settings used in our live tests are not the production settings. In test mode the model decides and writes every turn, and the safety checks only note what they would have stopped, they do not block. That is why invented praise reached callers in tests. In test mode the model also writes the booking read back and confirmation in its own words. In the production setting, the booking questions, read back and confirmation are spoken by code, word for word, and the checks block. So when you judge quality from a live test call, remember it is the looser mode.


## Why each technology

**[38:32] Reasons that are recorded**

Why each technology? I will give the reasons that are recorded and say where none is. Pipecat gives us the streaming audio pipeline and interruption handling, and ships with the speech detector and the turn model. Smart Turn listens to how a sentence ends, so a pause is not taken as the end. The production plan uses vLLM for the model and Whisper because it batches and caches prompt prefixes, and an earlier measurement showed more slots in Ollama did not help. The Qwen3 recogniser has about half the word errors of Whisper base, eight percent against sixteen, and can be given the car names. It is Mac only. Pocket TTS had the quickest first sound of the voices tried, about a tenth of a second, and a natural voice. Bedrock with Nemotron was the fastest of eleven models that invented nothing, but is for development only.

**[39:21] Reasons given by your boss, and defaults**

Redis was your boss's choice, so call content stays out of the stock and bookings database, and entries expire by themselves. Prometheus, the metrics adapter and the autoscaler are the standard way to scale on a custom number, and ours, demand including refused callers, is not a standard processor figure. For Python, FastAPI and Kubernetes the notes do not record a head to head comparison. They are common defaults that fit. The choices backed by measurements are the recogniser, the voice, the model, the speech detector and Smart Turn.


## Where things stand

**[39:54] What is done and committed**

Where do things stand? All the work is committed on the feature branch, not pushed. That includes the latency changes and call memory, the capacity and load tests with the stand in services and the soak test, the user journeys script, the simulation up to a thousand callers, the Redis default, the advise tool, the autoscaling walkthrough, and the notes, sections ninety two to one hundred and nine. The full test suite passes, one thousand eight hundred and eighty nine tests. Three report files are still uncommitted, as you asked: the capacity report, the soak report and the voice call walkthrough.

**[40:28] What is not proven**

What is not proven? A thousand people all talking, which needs rented machines. The deployment files on a real Kubernetes cluster. The production recogniser and model numbers, because ours are from a Mac and from Bedrock. Redis at a thousand or more calls, or across machines. The model deciding setting with many callers, and how often the advise tool picks the right need. These are the honest open items, and the next real step is the rented machine test with a staged ramp.

**[40:57] Thank you**

That is everything so far. If any part was unclear, tell me which chapter, and I can go deeper on it or make a shorter video on just that part. Every number in this video comes from the notes in the project, in the architecture review document. Thank you for listening.
