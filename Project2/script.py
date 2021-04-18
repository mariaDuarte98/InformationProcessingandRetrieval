import os
import shutil

rel_docs = [['39496', '46547', '46974', '62325', '63261', '82330', '82454'], ['3265', '3827', '3828', '3833', '3834', '3835', '3837', '3976', '4306', '4310', '4358', '4395', '4439', '4881', '5862', '6497', '6498', '6503', '6534', '6636', '6735', '7115', '7118', '7502', '8327', '8333', '8455', '9703', '9790', '11481', '11485', '11922', '11923', '11930', '11979', '12479', '12812', '13313', '13314', '13320', '13335', '13342', '13345', '14707', '14713', '15077', '15082', '15200', '16897', '16899', '16922', '17517', '17587', '19033', '19599', '19601', '19615', '20127', '20133', '20160', '20201', '20688', '20766', '21895', '22690', '23377', '24485', '24490', '24515', '25096', '25100', '25699', '26611', '27384', '27870', '28657', '28662', '29894', '29904', '29906', '29908', '29911', '29920', '30456', '30486', '33172', '33203', '33668', '35396', '36447', '38442', '39349', '44163', '46736', '47551', '49815', '49884', '51210', '53469', '53502', '53523', '53682', '54745', '58476', '58755', '61689', '64626', '64645', '64657', '65036', '65306', '65364', '65394', '65446', '66107', '67280', '68419', '68909', '70668', '71001', '71368', '73038', '73598', '73614', '73641', '73706', '73762', '75563', '76635', '76766', '77310', '78836', '79021', '81731', '82227'], ['6062', '16391', '27106', '27426', '59459', '61141', '66866', '72468', '78569', '78586', '81463', '82912', '83370', '84873'], ['3827', '3828', '3833', '3834', '3835', '3837', '3976', '4306', '4310', '4395', '4439', '4881', '5862', '6497', '6503', '6534', '6636', '7118', '7502', '8327', '8333', '8455', '9703', '9790', '11481', '11485', '11923', '11930', '11979', '12479', '12812', '12825', '13313', '13314', '13320', '13332', '13335', '13342', '13345', '14707', '14713', '15077', '15082', '15200', '16897', '16899', '16922', '16954', '17517', '17587', '17619', '19599', '19601', '19615', '19634', '20127', '20133', '20201', '20688', '20766', '21895', '22690', '23377', '24485', '24490', '25096', '25100', '25103', '25205', '25699', '26598', '26611', '27384', '27870', '28662', '29894', '29904', '29906', '29908', '29911', '29913', '29935', '30486', '33172', '33203', '33668', '35396', '36447', '38442', '39349', '44163', '44237', '44758', '44865', '47551', '49815', '51210', '53515', '53721', '54745', '54762', '56149', '56829', '58755', '64626', '65364', '65394', '67280', '68419', '70668', '71001', '73598', '73614', '73706', '73746', '73762', '75949', '76690', '76766', '79021'], ['2493', '2494', '3008', '5004', '5225', '5226', '15744', '40239', '40259', '48148', '49633', '51493', '80484', '80884', '86042', '86961'], ['6409', '16723', '16841', '41355'], ['52237', '77936', '79950'], ['10810', '19961', '84420'], ['16953', '24340', '26073', '31530', '46566', '56735', '58428', '58582', '58651', '58676', '59340', '61540', '62293', '64476', '65289', '67144', '67717', '68812', '78626', '82229'], ['38356', '41791', '42439', '82926', '85147'], ['45422', '46485', '46968'], ['59464', '70466', '78115', '78269', '83201', '85553'], ['12556', '12557', '13685', '13690', '17030', '17217', '24251', '25500', '37815', '40234', '79404', '86209'], ['18879', '21110', '25303', '38452', '60810'], ['9611', '61647', '76605'], ['13248', '22679', '24911', '26903', '49463', '49575', '53140', '60110', '67922', '68716', '68718', '70303', '74010', '78749', '81084', '83291'], ['60811', '62586', '71102'], ['20763', '43173', '50269'], ['26074', '63311', '64393', '67199'], ['3149', '4527', '24062', '43393', '48981', '49288', '49328', '62781', '75135'], ['9384', '13294', '13309', '14439', '14556', '14772', '15074', '15410', '15415', '17004', '29645', '34792', '75560', '79604'], ['38244', '39922', '44818', '53720', '54175', '54747', '61734', '62470', '69461', '72052', '73186', '76564', '78866', '86159', '86296'], ['2454', '10515', '37440'], ['7225', '9981', '19285', '74036', '74846', '79951'], ['18877', '27974', '48375', '68664', '77381', '77907', '79949', '80922', '82085', '82106', '82519', '86448'], ['6940', '14933', '19953', '22250', '38191', '40554', '56499', '59083', '60588', '63280', '67522', '69725', '71223', '74632', '75770', '76216', '78775', '78928', '86111'], ['26026', '26144', '40292', '67104', '75738'], ['3740', '9846', '37136', '56170'], ['16727', '21791', '35148', '46494', '46509', '58611', '72826', '72899', '73475', '75465', '75487', '75590', '75593', '78782', '83312', '83339', '83958'], ['15776', '18235', '46458'], ['26568', '40119', '41857', '41975'], ['4017', '7486', '67115', '70329', '71676', '71710', '77086'], ['35853', '54844', '74681', '74695', '84398'], ['12892', '73337', '73372', '73397', '74790'], ['31784', '40131', '41556', '44221', '59518', '65311', '65465', '65501', '66118', '66181', '67275', '67433', '82689', '86691'], ['15404', '35104', '43962', '46422', '58438', '61191', '72605', '76779'], ['61252', '66879', '85322'], ['2552', '21088', '48174', '50408', '74775', '75463', '77921'], ['62586', '67204', '71102'], ['11825', '18151', '18393', '22321', '25063', '43139', '44400', '45396', '58343', '80790', '82249'], ['17505', '19480', '27814', '35053', '39304', '39864', '39883', '50327', '52215', '55848', '58503', '58514', '60155', '60221', '62212', '62213', '62292', '65284', '75471', '75641', '76612', '78720', '81592', '83965'], ['6158', '28354', '45201', '50474'], ['5185', '5189', '72032', '76550'], ['27525', '27535', '30016', '35258', '36318', '69511'], ['7057', '17371', '27378', '45707', '82244'], ['8506', '11944', '12035', '12476', '14541', '19416', '19877', '29668', '29887', '29889', '29909', '32351', '83863'], ['27441', '27911', '34675', '45740', '64153', '73146'], ['52661', '55609', '58199', '60293', '66880', '66895', '68709', '75205', '83678', '85344', '85663', '85759'], ['17360', '17390', '30390', '34498', '41792'], ['19420', '19885', '53361', '54538'], ['32076', '48181', '54489', '60066', '70962', '83628'], ['29565', '31761', '78613', '85147', '86014'], ['12053', '12189', '30439', '39987', '42724', '45282', '46328', '49678', '77558', '80013'], ['37457', '37585', '37603', '37933', '38060', '39776'], ['4940', '4948', '19765', '24847', '24848', '24849', '38222', '40113', '67112', '78709', '79438'], ['37811', '53047', '53585', '60503', '65020', '77933'], ['53260', '55057', '69472'], ['20889', '44786', '51607', '73722', '73849'], ['3401', '4754', '5647', '14355', '26334', '27427', '27841', '28426', '37983', '41067', '45414', '46978', '47161', '49211', '62434', '64586', '64910', '81562', '86464', '86480', '86793'], ['8510', '17426', '19936', '24624', '25019', '26040', '28498', '32049', '37337', '39999', '41483', '51356', '60218', '68810', '78717'], ['10810', '11568', '40733', '51234', '63370'], ['42020', '55340', '73623', '75868', '75956', '84623'], ['35363', '61766', '67957', '73792'], ['3160', '4974', '5198', '6423', '19218', '31456', '39863', '39876', '42534', '43911', '44384', '47473', '48628', '49532', '50298', '58211', '68314', '72876', '73014', '79037', '86201'], ['56318', '62099', '63598', '63600', '63623', '68052', '68057'], ['5874', '29443', '31687', '31741', '54548', '64016', '80295', '81345'], ['44445', '63805', '65485', '71528', '78148'], ['4542', '14365', '14430', '16513', '23035', '25094', '27827', '28422', '31406', '33885', '35405', '37010', '37128', '37712', '39332', '41532', '42035', '42575', '49058', '49214', '50032', '53847', '58747', '62329', '63038', '73581', '76719', '76804', '78324', '78402', '79873', '84802'], ['24518', '25212', '33618', '71056', '80908'], ['16274', '24578', '26334', '37126', '45812', '46629', '49541', '49559', '51099', '56817', '58193', '59038', '67006', '67275', '77177', '81562'], ['10930', '22057', '37572', '52667', '56197', '58748', '69290'], ['27582', '28508', '33896', '48349', '60177', '60282', '71884', '72880', '82543', '85666'], ['3474', '6325', '7952', '8701', '14840', '21113', '28031', '29728', '31122', '32072', '35318', '38336', '41719', '41742', '46441', '50497', '52639', '53396', '60066', '60199', '61476', '66321', '73569', '73739', '77921', '84364', '86333'], ['9337', '18037', '26254', '69430', '77939'], ['3169', '10117', '11765', '13806', '16236', '17930', '18297', '18529', '22500', '23448', '28185', '31761', '36212', '37330', '41791', '42439', '45161', '45277', '46106', '47512', '58209', '61532', '66686', '67673', '69377', '70026', '71157', '71159', '72535', '75282', '77936', '78164', '81367', '81432', '83232', '85147', '86286'], ['40799', '42548', '44771', '44915', '44940'], ['8510', '19936', '26693', '26694', '26695', '27580', '27629', '27632', '27725', '28498', '30587', '30743', '32049', '33383', '37337', '39999', '41694', '48178', '50353', '50354', '50526', '54139', '60218', '68810', '78717'], ['34952', '50954', '54152'], ['39902', '47713', '52764', '60177', '60282'], ['44771', '44795', '44831', '62434', '79093'], ['15601', '16158', '51343', '51456'], ['24777', '26675', '27721', '30739', '31215', '32813', '33466', '34882', '35703', '40169', '48213', '51286', '51532', '53571', '54275', '56264', '74364', '77962', '78401'], ['3799', '11389', '20008', '28389', '39880', '40306', '41345', '41402', '44564', '46570', '49582', '50587', '53269', '54137', '54142', '54153', '61402', '62308', '65357', '68377', '70437', '70783', '70785', '71183', '83759'], ['58886', '59571', '60133', '60134', '60684', '72679', '75281', '76735', '77316'], ['7073', '19641', '21327', '22821', '27445', '29938', '31700', '39920', '45681', '45949', '46819', '47584', '47602', '49081', '49108', '51826', '52567', '55212', '55412', '56866', '56883', '63820', '63905', '71262', '76654', '76751'], ['21902', '26851', '41682', '42792', '43328', '44451', '46814', '47291', '48829', '49860', '49870', '49892', '68008', '68050', '69477', '70692', '70693', '70714', '73661', '79830'], ['4478', '9612', '29611', '31968', '69388', '75234', '86193'], ['9693', '31404', '68342'], ['7143', '12047', '18037', '20124', '34543', '34547', '38182', '50927', '52053', '52098', '55220', '72627'], ['15577', '22928', '27979', '51396', '52754', '52764', '57030', '57804', '60201', '65918', '81248', '81256', '81275'], ['8020', '12019', '15185', '15186', '63395'], ['40583', '41376', '57087'], ['9968', '37602', '46923', '46928', '46977'], ['21863', '22274', '28522', '31208', '31214', '32358', '33208', '33222', '33227', '33602', '33617', '37558', '49717', '56110', '57757', '59164', '60156', '60609', '62335', '62986', '64637', '64708', '64710', '64884', '65349', '65445', '65649', '72538', '76347', '82151', '85972'], ['3805', '5546', '6329', '14661', '43886', '50954', '78871', '83902'], ['2710', '16469', '40718', '68881', '82112'], ['5339', '12035', '12660', '15902', '21580', '22398', '32375', '37247', '46255', '51310', '54546', '63261', '64531', '66542', '67532', '69472', '70111', '72245', '72802', '75043', '75706', '81475'], ['6096', '16058', '53276'], ['3168', '6899', '15884', '16573', '24366', '33182', '34268', '38663', '39008', '42231', '48994', '49729', '50726', '51555', '58947', '62828', '63126', '81157', '81195', '81201', '84958'], ['19548', '21067', '28299', '31561', '42989', '71003', '71346']]

folders = os.listdir("rcv1_train/")
current = os.getcwd() 
topics = 5

for folder in folders:
    print("hvfgbvdsfgnhg")
    if os.path.isdir("rcv1_train/" + folder + "/"):
        xml = os.listdir("rcv1_train/" + folder + "/")
        for fich in xml:
            print("\n--------" + fich + "----------\n")
            name = fich.replace("newsML.xml","")
            print('nameeeeeeee')
            print(name)
            for i in range(topics):
                if not os.path.isdir("./rcv1"):
                    os.mkdir("rcv1")
                if  not os.path.isdir("./rcv1/rcv1_rel" + str(topics) + "/") and name in rel_docs[i]:
                    os.mkdir("rcv1/rcv1_rel" + str(topics) + "/")
                    print("\nnotexists" + name + "\n")
                    shutil.copy("rcv1_train/" + folder + "/" + fich, "rcv1/rcv1_rel" +  str(topics) + "/")
                elif os.path.isdir("./rcv1/rcv1_rel"  + str(topics) + "/")  and name in rel_docs[i]:
                    print("\n" + name + "\n")
                    shutil.copy("rcv1_train/" + folder + "/" + fich, "rcv1/rcv1_rel"  + str(topics) + "/")