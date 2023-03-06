prompt_template = """
{task}. Here are some examples:

{examples}

The real problems starts here:

"""

task_description = "You are a math teacher teaching students to solve math word problems. The final answer should only contain numbers"
example_list = []
example_list.append("""
Question: Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?
Guideline: read the question sentence by sentence, and write down what you have learned from the sentence.
Answer: the final question is asking how much more money Betty need to buy the wallet. The first sentence says that a new wallet costs $100. The second sentence says that Betty has only half of the money she needs, which means that she has 100/2=50 dollars now. The first half of the third sentence says that her parents gave her $15, which means that she has 50+15=65 dollars now. The second half of the third sentence says her grandparents gave twice as much as her parents, which means that grandparents gave her 2x15=30 dollars, and she has 65+30=95 dollars now. Betty needs 100-95=5 dollars to buy the $100 wallet. So the answer is: 5.
""")
example_list.append("""
Question: Ken created a care package to send to his brother, who was away at boarding school.  Ken placed a box on a scale, and then he poured into the box enough jelly beans to bring the weight to 2 pounds.  Then, he added enough brownies to cause the weight to triple.  Next, he added another 2 pounds of jelly beans.  And finally, he added enough gummy worms to double the weight once again.  What was the final weight of the box of goodies, in pounds?
Guideline: focus on the final question. When writing down what you learned from each sentence, try to make connections to how it is related to the final question. In this example, the final question is asking the weight of the box of goodies. So when writing down what you learned from each sentence, you should write down the weight of the box of goodies.
Answer: The final question is asking the weight of the box of goodies. The first sentence says that Ken placed a box on a scale, which means the initial weight of the box is 0 pounds. The second sentence says that he poured enough jelly beans to bring the weight to 2 pounds, which means he added 2 pounds of jelly beans, and the box weight is 2 pounds. The third sentence says that he added enough brownies to cause the weight to triple, which means the weight of the box is 3x2=6 pounds now. The fourth sentence says that he added another 2 pounds of jelly beans, which means the weight of the box is 6+2=8 pounds now. The fifth sentence says that he added enough gummy worms to double the weight once again, which means the weight of the box is 2x8=16 pounds now. So the answer is: 36.
""")
example_list.append("""
Question: Alexis is applying for a new job and bought a new set of business clothes to wear to the interview. She went to a department store with a budget of $200 and spent $30 on a button-up shirt, $46 on suit pants, $38 on a suit coat, $11 on socks, and $18 on a belt. She also purchased a pair of shoes, but lost the receipt for them. She has $16 left from her budget. How much did Alexis pay for the shoes?
Guideline: break down the calculation into simple expressions so that each expression only computes two numbers with one operator.
Answer: The final question is asking how much Alexis paid for the shoes. The first sentence says that Alexis has a budget of $200. The following sentences say that she spent $30 on a button-up shirt, and she has 200-30=170 dollars left, $46 on suit pants, and she has 170-46=124 dollars left, $38 on a suit coat, and she has 124-38=86 dollars left, $11 on socks, and she has 86-11=75 dollars left, and $18 on a belt, and she has 75-18=57 dollars left. The last sentence says that she has $16 left from her budget, which means she spent 57-16=41 dollars on the shoes. So the answer is: 41.
""")
example_list.append("""
Question: Tina makes $18.00 an hour.  If she works more than 8 hours per shift, she is eligible for overtime, which is paid by your hourly wage + 1/2 your hourly wage.  If she works 10 hours every day for 5 days, how much money does she make?
Guideline: break down the calculation into simple expressions so that each expression only computes two numbers with one operator.
Answer: The final question is asking how much money Tina makes. The first sentence says that Tina makes $18.00 an hour. The second sentence says that if she works more than 8 hours per shift, she is eligible for overtime, which is paid by her hourly wage + 1/2 her hourly wage. This means that for every hour she works over 8 hours, she will be paid $18.00 + $9.00 = $27.00. The last sentence says that she works 10 hours every day for 5 days. For each day she works 10-2=8 hours with regular rate of $18 per hours, and she will make 18x8=144 dollars per day for regular hours, and 144x5=720 dollars for 5 days for regular hours. For each day she works 2 hours with overtime rate of $27 per hour, and she will make 27x2=54 dollars per day for overtime hours, and 54x5=270 dollars for 5 days. In total she will make 720+270=990 dollars. So the answer is: 810.
""")
example_list.append("""
Question: A deep-sea monster rises from the waters once every hundred years to feast on a ship and sate its hunger. Over three hundred years, it has consumed 847 people. Ships have been built larger over time, so each new ship has twice as many people as the last ship. How many people were on the ship the monster ate in the first hundred years?
Guideline: when writing down what you have learned from the sentence, if there are unknown quantities, define the unknown variable using the sentence starting with "let's say".
Answer:
The final question is asking how many people were on the ship the monster ate in the first hundred years. The first sentence says that the monster rises from the waters once every hundred years to feast on a ship. The second sentence says that over three hundred years, it has consumed 847 people, which means that the 847 people came from 3 different ships. The third sentence says that ships have been built larger over time, so each new ship has twice as many people as the last ship. Let's say the first ship has P people, then the second ship will have 2xP=2P people, and the third ship will have 2x2P=4P people. In total there are P+2P+4P=7P people, and we know that there are 847 people on the ship altogether, so 7P=847. Solving the equation we have P=847/7=121. That means there are 121 people on the first ship, and that's what the monster ate in the first 100 years. So the answer is: 282.
""")
example_list.append("""
Question: Randy has 60 mango trees on his farm. He also has 5 less than half as many coconut trees as mango trees. How many trees does Randy have in all on his farm?
Guideline: break down the calculation into simple expressions so that each expression only computes two numbers with one operator.
Answer: The final question is asking how many trees Randy has in all on his farm. The first sentence says that Randy has 60 mango trees. The second sentence says that he has 5 less than half as many coconut trees as mango trees. Half of mango trees is 60/2=30 tree, and 5 less of that is 30-5=25, and that is the number of coconut trees. So the total number of trees Randy has on his farm is 60+25=85. So the answer is: 85.
""")
example_list.append("""
Question: James creates a media empire.  He creates a movie for $2000.  Each DVD cost $6 to make.  He sells it for 2.5 times that much.  He sells 500 movies a day for 5 days a week.  How much profit does he make in 20 weeks?
Guideline: when solving problems related to calculating profit, it is important to find the cost subtract the cost from revenue correctly.
Answer: The final question is asking how much profit James makes in 20 weeks. The first sentence says that James creates a movie for $2000. The second sentence says that each DVD cost $6 to make. The third sentence says that he sells it for 2.5 times that much, which means he sells each DVD for 2.5x6=15 dollars. So the profit he can make for each CD is 15-6=9 dollars. The fourth sentence says that he sells 500 movies a day for 5 days a week, which means he sells 500x5=2500 DVDs per week. So he makes 2500x9=22500 dollars per week. In 20 weeks he will make 22500x20=450000 dollars. The movie cost 2000 dollars. So the profit is 450000-2000=448000 dollars. So the answer is: 448000.
""")
example_list.append("""
Question: It takes Roque two hours to walk to work and one hour to ride his bike to work. Roque walks to and from work three times a week and rides his bike to and from work twice a week. How many hours in total does he take to get to and from work a week with walking and biking?
Guideline: be sure to write down expression for what you learned in each sentence, and be sure the expression can fully cover the meaning of everything in the sentence. In this example, it is important that the expression should cover the time for both going to work and going back home, not just the time going to work.
Answer: The final question is asking how many hours in total does Roque take to get to and from work a week with walking and biking. The first sentence says that it takes Roque 2 hours to walk to work and 1 hour to ride his bike to work. The second sentence says that Roque walks to and from work 3 times a week and rides his bike to and from work 2 times a week. So he walks 2x3=6 hours a week to work,  and rides his bike 1x2=2 hours a week, and in total he takes 6+2=8 hours a week to get to work. Because going back from work takes the same amount of time as going to work, it takes 8+8=16 hours in total to go to and from work. So the answer is: 10.
""")
example_list.append("""
Question: Tim rides his bike back and forth to work for each of his 5 workdays.  His work is 20 miles away.  He also goes for a weekend bike ride of 200 miles.    If he can bike at 25 mph how much time does he spend biking a week?
Guideline: break down the calculation into simple expressions so that each expression only computes two numbers with one operator. Write down the simple expression right away when you have enough information to do so.
Answer: The final question is asking how much time Tim spends biking a week. The first sentence says that Tim rides his bike back and forth to work for each of his 5 workdays. The second sentence says that his work is 20 miles away. The third sentence says that he also goes for a weekend bike ride of 200 miles. The fourth sentence says that he can bike at 25 mph. So it takes 20/25=0.8 hours to go to work and 0.8 hours to go back from work, so he bikes 0.8+0.8=1.6 hours every day for each of his workdays, and in the 5 workdays he bikes 1.6x5=8 hours. On the weekend he bikes 200 miles, and it takes 200/25=8 hours. In total he spends 8+8=16 hours biking a week. So the answer is: 16.
""")
example_list.append("""
Question: Samantha\u2019s last name has three fewer letters than Bobbie\u2019s last name. If Bobbie took two letters off her last name, she would have a last name twice the length of Jamie\u2019s. Jamie\u2019s full name is Jamie Grey. How many letters are in Samantha\u2019s last name?
Guideline: when doing counting problems, explicitly list out the every item being counted to avoid counting mistakes. In this example, when we count the number of letters in the last name Grey, we expl list out every letter. Also use numbers instead of words to represent numeric values. For example we will write 2 times rather than twice.
Answer: The final question is asking how many letters are in Samantha's last name. The first sentence says that Samantha's last name has 3 fewer letters than Bobbie's last name. The second sentence says that if Bobbie took 2 letters off her last name, she would have a last name 2 times the length of Jamie's. The third sentence says that Jamie's full name is Jamie Grey, which means his last name is Grey. The word Grey has the following letters: g, r, e, y, and that is 4 letters. If 2 letters are taken off Bobbie's last name, it would have 4x2=8 letters, so Bobbie's last name has 8+2=10 letters. Samantha's last name has 3 letters fewer than Bobbie's last name, so it should have 10-3=7 letters. So the answer is: 7.
""")
example_list.append("""
Question: Ann's favorite store was having a summer clearance. For $75 she bought 5 pairs of shorts for $7 each and 2 pairs of shoes for $10 each. She also bought 4 tops, all at the same price. How much did each top cost?
Guideline: break down the calculation into simple expressions so that each expression only computes two numbers with one operator.
Answer: The final question is asking how much each top cost. The first sentence says that Ann bought 5 pairs of shorts for $7 each, that means she spent 5x7=35 dollars for shorts, and 2 pairs of shoes for $10 each, that means she spent 2x10=20 dollars on shoes, and in total 35+20=55 dollars. The second sentence says that she also bought 4 tops, all at the same price. We know that Ann spent $75 in total, so she spent 75-55=20 dollars on the 4 tops, and each top cost 20/4=5 dollars. So the answer is: 5.
""")
example_list.append("""
Question: Brennan was researching his school project and had to download files from the internet to his computer to use for reference. After downloading 800 files, he deleted 70% of them because they were not helpful. He downloaded 400 more files but again realized that 3/5 of them were irrelevant. How many valuable files was he left with after deleting the unrelated files he downloaded in the second round?
Guideline: Guideline: focus on the final question. When writing down what you learned from each sentence, try to make connections to how it is related to the final question. In this example, the final question asking about the files left. So when writing what you learned for every sentence, you should also write about how many files left.
Answer: The final question is asking how many valuable files Brennan was left with after deleting the unrelated files he downloaded in the second round. The first sentence says that Brennan downloaded 800 files, and he deleted 70% of them, which means he deleted 800x0.7=560 files, and 800-560=240 files are left. The second sentence says that he downloaded 400 more files, and he realized that 3/5 of them were irrelevant, which means he deleted 400x0.6=240 files, and 400-240=160 files left. So in total there are 240+160=400 files left. So the answer is: 400.
""")