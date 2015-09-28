"""
Plot the performances of the many different NMTF algorithms in a single graph,
as we vary the fraction of missing values

We plot the average performance across all 10 attempts for different fractions:
[0.1, 0.2, ..., 0.9].

We use a dataset of I=100, J=80, K=10, with unit mean priors and zero mean unit
variance noise.

We have the following methods:
- VB NMTF
- Gibbs NMTF

"""

import matplotlib.pyplot as plt
metrics = ['MSE','R^2','Rp']

fractions_unknown = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] 

# VB NMF
vb_all_performances = {'R^2': [[0.9980084354345304, 0.9975696566291119, 0.997899530358802, 0.9979873022927159, 0.998314813392778, 0.9981115849959178, 0.998204022010684, 0.9983657711527754, 0.9979535155178162, 0.9978393045076107], [0.997985242994016, 0.997707730689699, 0.9982142360088324, 0.9979010220906681, 0.9978761132788142, 0.9978390223744658, 0.9980422830661413, 0.9980903586230472, 0.9979507575136342, 0.9980108713682542], [0.9953143826215496, 0.997718685204668, 0.9976863854522392, 0.9978652846151879, 0.9980088932921369, 0.9978924043190982, 0.9979571951280339, 0.9980441162589905, 0.9978208468263103, 0.9976738478239459], [0.9976761679547769, 0.9977500018118746, 0.9978198310559514, 0.9978378944439149, 0.997868285153773, 0.9977716972983648, 0.9977281352335649, 0.9978408078364147, 0.997533244752905, 0.9978953730251335], [0.9974706750597508, 0.9958951164608203, 0.9977101426980319, 0.9976493783844218, 0.9978827257524434, 0.997519934775041, 0.9977304572719358, 0.9975875104869111, 0.9974912956479653, 0.9976145402305994], [0.9974926506148456, 0.9976383592912337, 0.997551567468177, 0.9971608824097004, 0.9973421701541971, 0.9974524972449568, 0.9974935574634191, 0.9974383763670657, 0.9976281216788087, 0.9965804518479184], [0.9862037180523757, 0.9970865009759344, 0.9960969774322391, 0.9970549395566566, 0.9968776359101204, 0.9970791826414068, 0.9969844140879247, 0.9972035742793113, 0.9969878634367444, 0.9968780018213315], [0.9918848272024572, 0.9926329987032696, 0.9922815700772748, 0.9916439279835809, 0.951064744887406, 0.992426885312417, 0.9894038788684064, 0.9898004355741431, 0.9929771884411486, 0.9943478062207849], [0.7888375587374407, 0.923859395370809, 0.8965454779737884, 0.904438700571149, 0.8935410897501692, 0.9039486402512202, 0.772343145239076, 0.9344033004288783, 0.9483545226650639, 0.9427069652200406]], 'MSE': [[1.1657843015314036, 1.1310264312047598, 1.2223933884230676, 1.2194568904698131, 1.2181050444775456, 1.1005962820855677, 1.227296493439513, 1.1133841310570984, 1.1627530893071527, 1.1577619958776113], [1.195273435888109, 1.1558838938845879, 1.1325208517608689, 1.1132528384785132, 1.1925364544466646, 1.1650604188232441, 1.145333660125863, 1.2459041366787149, 1.2217598113385657, 1.2285783809786903], [2.5227924825582084, 1.2187040812501313, 1.2295401245561275, 1.2278400118784973, 1.1921630037909197, 1.163844014686062, 1.1683603044548831, 1.23693715526466, 1.2677412854560537, 1.2819568703687134], [1.3019099036895097, 1.315319470092019, 1.297217944433692, 1.3094111425598021, 1.243927919333673, 1.2166287958265385, 1.3048596498757079, 1.2751673704572213, 1.3983616108263883, 1.2566468359715623], [1.3188811314147815, 2.2156788391299398, 1.2618372487711629, 1.3800704377973192, 1.2867518030425937, 1.3017204646978655, 1.3254823222857968, 1.3943802189930108, 1.3964001298919739, 1.3545849890924837], [1.3802457235468211, 1.3955976956225271, 1.4931094662619493, 1.6233262171830227, 1.5778245249896121, 1.4222466219223044, 1.4010534239591481, 1.4296867477636876, 1.3742549583682011, 2.0176526010472351], [8.0051279115674436, 1.7005854162671854, 2.2327948463791931, 1.6961089410217278, 1.7982260675258219, 1.7143000978733312, 1.7674601216584889, 1.616531013257517, 1.6935808671918777, 1.8473489668554552], [4.6537541047694422, 4.1694367208207872, 4.4534158849103855, 4.9636406897063328, 29.093237390496842, 4.2735356092313701, 6.0515062845854235, 5.8542290798232184, 4.1390441839779122, 3.2118990427263947], [121.10059265867221, 43.750676089249886, 56.574051156328437, 54.831485200977937, 62.291200861831186, 56.056900229527947, 130.96997335839021, 37.893151938137059, 29.246359276026695, 33.044971212327326]], 'Rp': [[0.99900399749945878, 0.99878933600809749, 0.99894936530079703, 0.99899940002245924, 0.99916106923500758, 0.99905651867630363, 0.99910273058028676, 0.99918697689777436, 0.99897723157355545, 0.9989232918807418], [0.99899232822094464, 0.9988535822849739, 0.99910813859304637, 0.99895064474570805, 0.9989376844059642, 0.99891953715820214, 0.99902275617211012, 0.99904605903809607, 0.99897542311717136, 0.99900685989376703], [0.99765795561964254, 0.99886087446031036, 0.99884809257564289, 0.99893233787408819, 0.99900419615914837, 0.99894677235265628, 0.99897812417900622, 0.99902584040418863, 0.99891380687762232, 0.9988368384009696], [0.99883998483596326, 0.99887441451844194, 0.99891028710166974, 0.99891921095205471, 0.99893435084780879, 0.99888621669132238, 0.99886486889303594, 0.99892255393098317, 0.99876672010598178, 0.99894738740340172], [0.99873484419303904, 0.99795078724560293, 0.99885484104626543, 0.99882729600954978, 0.99894201759051704, 0.99876556298770502, 0.99886511635018327, 0.99879366357620192, 0.99874665328143086, 0.99881420718405456], [0.99874623190589296, 0.998819668778968, 0.99877509105505291, 0.99858077957149138, 0.99867154636924294, 0.99872552964504857, 0.9987471550427991, 0.99872120107451767, 0.99881497807268804, 0.99828908162226848], [0.9930987926900251, 0.99854439082025503, 0.9980484896592754, 0.99852680184649212, 0.99844030059289968, 0.99853981755915155, 0.99849146957710933, 0.99860186332767265, 0.99849301439840887, 0.99845119966613416], [0.99594415010831405, 0.99631121342444728, 0.99614974105143772, 0.99581432149566163, 0.97754659680024791, 0.99621243289555028, 0.99472734799434614, 0.99491070694055106, 0.99649307295350409, 0.99717323993530005], [0.90055411762456339, 0.96276047666311892, 0.95186367694294538, 0.95176394359642202, 0.94572889565934992, 0.95087759320961696, 0.91564525300171662, 0.96775735024495457, 0.97488984841957393, 0.97093944398382537]]} 
vb_average_performances = {'R^2': [0.99802539362927423, 0.9979617638007573, 0.99759820415421596, 0.99777214385666757, 0.99745517767679215, 0.99737786345403223, 0.9958452808194046, 0.98784642632708886, 0.89089787962076361], 'MSE': [1.1718558047873529, 1.1796103882403821, 1.3509879334264254, 1.2919450643066113, 1.423578758511693, 1.5114997980664511, 2.4072064249598037, 7.0863698991048123, 62.575936198146906], 'Rp': [0.99901499176744823, 0.99898130136299845, 0.99880048389032738, 0.99888659952806624, 0.99872949894645502, 0.99868912631379703, 0.99792361401374219, 0.994128282359936, 0.94927805993460868]}

# Gibbs NMF
gibbs_all_performances = {'R^2': [[0.9982483427512238, 0.9982569106019503, 0.9980274882567254, 0.9981494402541832, 0.998278814922556, 0.9980119458152925, 0.9956195798944338, 0.9978656571576676, 0.9948160505407695, 0.9954424584311419], [0.9981333905047056, 0.9979100423069163, 0.9955442299110503, 0.9978676991250482, 0.9977577278512513, 0.9976874007674094, 0.9979539285118952, 0.9945971203469434, 0.9979930384329919, 0.9952512628300171], [0.9978378347855068, 0.9978801890932645, 0.9979429165098603, 0.9977768522969408, 0.9978015658203891, 0.9979260351113085, 0.996066582239531, 0.9978949112777393, 0.9957051056029234, 0.9978325348985365], [0.997676691750717, 0.9978240400104373, 0.9976582073437797, 0.9977758430041533, 0.9978329364443173, 0.9969584609777844, 0.9977265503307726, 0.9977142397526078, 0.9976826750311384, 0.9952749146733821], [0.9976399737704836, 0.99767873779128, 0.9976443573817393, 0.9977386446047206, 0.9951828897567764, 0.9951514756864354, 0.9975228041236924, 0.9946232273494868, 0.9978052124263244, 0.9975976092972867], [0.9974415991464385, 0.9973033769326197, 0.9974228772025158, 0.9946088115543904, 0.9952143307189987, 0.9973675266253234, 0.9973057070437303, 0.9975558402516823, 0.995458400143532, 0.9960985828203994], [0.9969530909108212, 0.9968029147445387, 0.9971504420281516, 0.9969488841655992, 0.9970958121880199, 0.9966773266250664, 0.9938682775624493, 0.9961049113779932, 0.9971622118352892, 0.9962192432064729], [0.9951026931828699, 0.7660836211471911, 0.9940445719835007, 0.9896536121295896, 0.9945659040952501, 0.988406717974716, 0.9945113008065473, 0.9933068784853915, 0.9948281657813091, 0.9883895464109994], [0.9604700232586909, 0.9418352745019882, 0.9369170279940039, 0.9565231890815216, 0.9358135672871284, 0.8901515028861764, 0.8526143726709253, 0.9208252572952669, 0.9356639631955865, 0.9150298363582962]], 'MSE': [[1.0808729277323119, 1.18368144459457, 1.2482719449362341, 1.1352963780276244, 1.1601753464233808, 1.2151973041786375, 2.4164737182848612, 1.0003148188857787, 2.5942229347788701, 2.5047896536227752], [1.1843314818742263, 1.1757671232787674, 2.3327245493978142, 1.180128107517465, 1.2842800506080199, 1.2049850235424111, 1.1741921882187809, 2.6600499717779962, 1.1793531712106733, 2.3543081940898762], [1.1934944009676398, 1.1532830073107854, 1.1881850407423593, 1.2723946725367128, 1.1969609359130404, 1.1820559488750182, 2.1844437217494734, 1.2357711054636829, 2.5090964813586178, 1.2489422401685326], [1.2567834776223799, 1.2821691826936066, 1.2778268649507891, 1.2398649211877735, 1.269188460871127, 1.7206999288175755, 1.2723402639850054, 1.3159164252553697, 1.3056890035508482, 2.6459275490234186], [1.3352877670730834, 1.3091042045422829, 1.3757174781757739, 1.3218586672018222, 2.8378647331470948, 2.8539780438341107, 1.3763994810692057, 3.1346541016602063, 1.2627770325484942, 1.3552280130779939], [1.4273094743725785, 1.434595115336204, 1.4414922183831713, 3.1444806421280123, 2.7650267118008376, 1.4994353926134174, 1.5055629844820049, 1.4048879955687767, 2.5038299916651563, 2.1756197439514939], [1.7057462912279295, 1.7687744319713692, 1.6912811269774135, 1.7545816423023897, 1.6997773173924611, 1.8524989169496267, 3.7351769734582869, 2.2455798397048525, 1.6333754736408459, 2.1284626338691859], [2.8764183132345198, 133.28868633880313, 3.4275049923166159, 5.8175482017201547, 3.0216424960900752, 6.636517556428573, 3.1689426347781566, 3.8465728232748919, 2.8425068991088369, 6.7790560467166632], [22.746691909833263, 33.965778984081581, 36.678462115333879, 24.265612382882104, 36.69617505247831, 63.349738963708447, 87.175153541863423, 44.434758152770009, 36.98431698415704, 48.325280731695656]], 'Rp': [[0.99912516809597618, 0.99912851773747435, 0.99902078016367113, 0.99907517480901609, 0.99914130584730865, 0.99900909539166127, 0.99781355387802606, 0.99893498768325473, 0.99741192949954183, 0.99772840388063777], [0.99906896273817913, 0.99895546524041756, 0.99777108816072069, 0.99893367385739462, 0.99888001792878434, 0.99884364370922263, 0.99898235904188715, 0.99729948561669479, 0.99899846939967685, 0.9976235544251435], [0.99891970066134772, 0.99894025119643926, 0.9989709990498068, 0.99889600199496409, 0.99890045003686645, 0.99896950839660326, 0.99803167530417058, 0.99894693152792535, 0.99785317200816015, 0.9989163680880413], [0.99883802086280249, 0.99891205124831806, 0.99882855861506181, 0.99888819899015546, 0.99891625616289104, 0.99847935187870773, 0.9988635008728064, 0.99885666071349033, 0.99884142021146249, 0.99763650528105718], [0.99881931461007267, 0.99883907225947521, 0.99882521703774507, 0.99886957062578896, 0.99759178641050417, 0.9975759499631629, 0.99876171369078648, 0.99731636912507449, 0.99890200604899682, 0.99879909022956881], [0.99872016035090416, 0.99865174508570609, 0.99871202743283161, 0.99730266352461239, 0.99761485691080254, 0.9986837676833864, 0.99865496163849943, 0.9987830854960299, 0.99772732112716123, 0.99805260696793585], [0.99847543269100858, 0.99840238958422556, 0.9985774179542497, 0.99847394537416589, 0.99854878675140835, 0.998342381253425, 0.99693604454918139, 0.99805221003147171, 0.99858452038867496, 0.99815566028782421], [0.99754896717769803, 0.90215350634469194, 0.99703268682471291, 0.99481732988659644, 0.99728421301420844, 0.99420108823119557, 0.99726133375675019, 0.99672051769687564, 0.99746222771838566, 0.99424539714288396], [0.98013962371773777, 0.97052142905834449, 0.96830971738355365, 0.97902496448688414, 0.96872344339992877, 0.9472304589893894, 0.93174801167995647, 0.96359386301586958, 0.96929248236658549, 0.96151655640683065]]} 
gibbs_average_performances = {'R^2': [0.99727166886259455, 0.99706958405882296, 0.99746645276360013, 0.99741245593190886, 0.99685849321882247, 0.99657770524396305, 0.99649831146444023, 0.96988930119973649, 0.9245844014529585], 'MSE': [1.5539296471465043, 1.573011986151603, 1.4364627555085865, 1.4586406077957892, 1.816286952233007, 1.9302240270301652, 2.0215254647494367, 17.170539630247159, 43.462196881880374], 'Rp': [0.99863889169865683, 0.99853567201181215, 0.99873450582643242, 0.99870605248367528, 0.99843000900011758, 0.99829031962178694, 0.9982548788865635, 0.98687272677939963, 0.96401005505050819]}



# Assemble the average performances and method names
methods = ['VB-NMF','G-NMF']
avr_performances = [
    vb_average_performances,
    gibbs_average_performances
]

for metric in metrics:
    plt.figure()
    #plt.title("Performances (%s) for different fractions of missing values" % metric)
    plt.xlabel("Fraction missing", fontsize=16)
    plt.ylabel(metric, fontsize=16)
    
    x = fractions_unknown
    for method, avr_performance in zip(methods,avr_performances):
        y = avr_performance[metric]
        #plt.plot(x,y,label=method)
        plt.plot(x,y,linestyle='-', marker='o', label=method)
    plt.legend(loc=0)  