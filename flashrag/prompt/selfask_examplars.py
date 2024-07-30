SELF_ASK_PROMPT_SINGLE_HOP = """Given the following question, answer it by providing follow up questions and intermediate answers. If intermediate questions are not necessarry, answer the question directly. You are provided with evidence that can help you arrive at the answer before the question.
#
Context1: The Big Red One: Fuller was a World War II veteran and served with the 1st Infantry Division, which is nicknamed "The Big Red One" for the red numeral "1" on the division's shoulder patch. He received the Silver Star, Bronze Star, and Purple Heart during his service.
Question: how did the big red one get its name
Are follow up questions needed here: No.
So the final answer is: its shoulder patch
#
Context1: Module:Location map/data/Cayman Islands: Module:Location map/data/Cayman Islands is a location map definition used to overlay markers and labels on an equirectangular projection map of Cayman
Question: where are the cayman islands on the map
Are follow up questions needed here: No.
So the final answer is: western Caribbean Sea
#
Context1: Korean War | Combatants, Summary, Years, Map ... - Britannica: After more than a million combat casualties had been suffered on both sides, the fighting ended in July 1953 with Korea still divided into two hostile states. Negotiations in 1954 produced no further agreement, and the front line has been accepted ever since as the de facto boundary between North and South Korea.
Question: who won the war between north korea and south korea
Are follow up questions needed here: No.
So the final answer is: technically still at war
#
Context1: It's Always Sunny in Philadelphia (season 13): The thirteenth season of the American comedy television series It's Always Sunny in Philadelphia premiered on FXX on September 5, 2018.
Question: when does it's always sunny in philadelphia season 13 start
Are follow up questions needed here: No.
So the final answer is: September 5, 2018
#
Context1: You've Got a Friend in Me: "You've Got a Friend in Me" is a song by Randy Newman. Used as the theme song for the 1995 Disney/Pixar animated film Toy Story, it has since become a major ...
Question: who sang you got a friend in me from toy story
Are follow up questions needed here: No.
So the final answer is: Randy Newman
#
Context1: Timeline of space exploration: This is a timeline of space exploration which includes notable achievements, first accomplishments and milestones in humanity's exploration of outer space.
Question: when was the first person sent to space
Are follow up questions needed here: No.
So the final answer is: 12 April 1961
#"""


SELF_ASK_PROMPT_MULTI_HOP = """Given the following question, answer it by providing follow up questions and intermediate answers. If intermediate questions are not necessarry, answer the question directly. You are provided with evidence that can help you arrive at the answer before the question.
#
Context1: Xawery Żuławski: Polish-Russian War (Wojna polsko-ruska) is a 2009 Polish film directed by Xawery Żuławski based on the novel Polish-Russian War under the white-red flag by Dorota Masłowska. So the answer is Xawery Żuławski.
Context2: Xawery Żuławski: Xawery Żuławski ; National Film School in Łódź · 1995–present · Maria Strzelecka · 2.
Question: Who is the mother of the director of film Polish-Russian War (Film)?
Are follow up questions needed here: Yes.
Follow up: Who is the director of the film Polish-Russian War (Film)?
Intermediate answer: The director of the film Polish-Russian War is Xawery Żuławski.
Follow up: Who is the mother of Xawery Żuławski?
Intermediate answer: The mother of Xawery Żuławski is Małgorzata Braunek.
So the final answer is: Rick Scott Małgorzata Braunek.
#
Context1: 2003: Blind Shaft (Chinese: 盲井; pinyin: Mángjǐng) is a 2003 film about a pair of brutal con artists operating in the illegal coal mines of present-day northern China. So the answer is 2003.
Context2: December 2, 1932: Release and reception. The Mask of Fu Manchu opened in New York on December 2, 1932. The film cost a total of $338,000 and had worldwide rentals of $625,000. It had a profit of $62,000. So the answer is December 2, 1932.
Question: Which film came out first, Blind Shaft or The Mask Of Fu Manchu?
Are follow up questions needed here: Yes.
Follow up: When did Blind Shaft come out?
Intermediate answer: Blind Shaft came out in 2003.
Follow up: When did The Mask Of Fu Manchu come out?
Intermediate answer: The Mask Of Fu Manchu came out in 1932.
So the final answer is: The Mask Of Fu Manchu.
#
Context1: John V, Prince of Anhalt-Zerbst: John was the second (but eldest surviving) son of Ernest I, Prince of Anhalt-Dessau, by his wife Margarete, daughter of Henry I, Duke of Münsterberg-Oels, and granddaughter of George of Poděbrady, King of Bohemia.
Context2: 12 June 1516: Ernest I, Prince of Anhalt-Dessau (died Dessau, 12 June 1516), was a German prince of the House of Ascania and ruler of the principality of Anhalt-Dessau. So the answer is 12 June 1516.
Question: When did John V, Prince Of Anhalt-Zerbst's father die?
Are follow up questions needed here: Yes.
Follow up: Who is the father of John V, Prince Of Anhalt-Zerbst?
Intermediate answer: The father of John V, Prince Of Anhalt-Zerbst is Ernest I, Prince of Anhalt-Dessau.
Follow up: When did Ernest I, Prince of Anhalt-Dessau die?
Intermediate answer: Ernest I, Prince of Anhalt-Dessau died on 12 June 1516.
So the final answer is: 12 June 1516
#
Context1: El extraño viaje: El extraño viaje (English: The Strange Voyage) is a 1964 Spanish black drama film directed by Fernando Fernán Gómez.
Context2: Love in Pawn: Love in Pawn is a 1953 British comedy film directed by Charles Saunders and starring Bernard Braden, Barbara Kelly and Jeannie Carson.
Context3: 28 August 1921: Fernando Fernández Gómez (28 August 1921 – 21 November 2007) better known as Fernando Fernán Gómez was a Spanish actor, screenwriter, film director, theater director and member of the Royal Spanish Academy for seven years. So the answer is 28 August 1921.
Context4: Charles Saunders (director): Charles Joel Saunders (8 April 1904 – 20 April 1997) was an English film director and screenwriter who began in the industry as a film editor, and who also contributed to television.
Question: Which film has the director who was born later, El Extraño Viaje or Love In Pawn?
Are follow up questions needed here: Yes.
Follow up: Who is the director of El Extraño Viaje?
Intermediate answer: The director of El Extraño Viaje is Fernando Fernán Gómez.
Follow up: Who is the director of Love in Pawn?
Intermediate answer: The director of Love in Pawn is Charles Saunders.
Follow up: When was Fernando Fernán Gómez born?
Intermediate answer: Fernando Fernán Gómez was born on 28 August 1921.
Follow up: When was Charles Saunders (director) born?
Intermediate answer: Charles Saunders was born on 8 April 1904.
So the final answer is: El Extraño Viaje.
#
Context1: John, Count Palatine of Neumarkt: John (Johann von Pfalz-Neumarkt; 1383 – 14 March 1443) was the Count Palatine of Neumarkt from 1410 to his death. The son of Rupert III of the Palatinate, he married Catherine of Pomerania in 1407.
Context2: John, Count Palatine of Neumarkt: John (Johann von Pfalz-Neumarkt; 1383 – 14 March 1443) was the Count Palatine of Neumarkt from 1410 to his death. The son of Rupert III of the Palatinate, he married Catherine of Pomerania in 1407.
Question: Who is Catherine Of Pomerania, Countess Palatine Of Neumarkt's father-in-law?
Are follow up questions needed here: Yes.
Follow up: Who is the husband of Catherine of Pomerania, Countess Palatine of Neumarkt?
Intermediate answer: The husband of Catherine of Pomerania, Countess Palatine of Neumarkt is John, Count Palatine of Neumarkt.
Follow up: Who is the father of John, Count Palatine of Neumarkt?
Intermediate answer: The father of John, Count Palatine of Neumarkt is Rupert III of the Palatinate.
So the final answer is: Rupert III of the Palatinate.
#
Context1: Crimen a las tres: Crimen a las tres is a 1935 Argentine crime film directed and written by Luis Saslavsky. Crimen a las tres. Directed by, Luis Saslavsky.
Context2: Elio Petri: The Working Class Goes to Heaven (Italian: La classe operaia va in paradiso), released in the US as Lulu the Tool, is a 1971 political drama film directed by Elio Petri. So the answer is Elio Petri.
Context3: March 20, 1995: Luis Saslavsky (April 21, 1903 – March 20, 1995) was an Argentine film director, screenwriter and film producer, and one of the influential directors in the Cinema of Argentina of the classic era. So the answer is March 20, 1995.
Context4: Elio Petri: Final years. In 1981, Petri visited Geneva to direct Arthur Miller\'s new play The American Clock, with Marcello Mastroianni playing the lead role. Petri died of cancer on 10 November 1982. He was 53 years old.
Question: Which film has the director died first, Crimen A Las Tres or The Working Class Goes To Heaven?
Are follow up questions needed here: Yes.
Follow up: Who is the director of Crimen a las tres?
Intermediate answer: The director of Crimen a las tres is Luis Saslavsky.
Follow up: Who is the director of The Working Class Goes to Heaven?
Intermediate answer: The director of The Working Class Goes to Heaven is Elio Petri.
Follow up: When did Luis Saslavsky die?
Intermediate answer: Luis Saslavsky died on March 20, 1995.
Follow up: When did Elio Petri die?
Intermediate answer: Elio Petri died on 10 November 1982.
So the final answer is: The Working Class Goes to Heaven
#"""
