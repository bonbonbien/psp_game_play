# N_QNS = 18
LEVEL = list(range(23))
LEVEL_GROUP = ["0-4", "5-12", "13-22"]
LVGP_ORDER = {"0-4": 0, "5-12": 1, "13-22": 2}
QNS_PER_LV_GP = {"0-4": list(range(1, 4)), "5-12": list(range(4, 14)), "13-22": list(range(14, 19))}
LV_PER_LV_GP = {"0-4": list(range(0, 5)), "5-12": list(range(5, 13)), "13-22": list(range(13, 23))}
CAT_FEAT_SIZE = {
    "event_comb_code": 19,
    "room_fqid_code": 19,
    "page_code": 8,
    "text_fqid_code": 127,
    "level_code": 23,
}


class CFG:
    def __init__(self, cur_grp='5-12'):
        self.INPUT_PATH = "../input/how-to-get-32gb-ram/train.parquet"
        self.TARGET_PATH = "../input/how-to-get-32gb-ram/train_labels.parquet"

        # Questions in a level-group, one model per group
        self.LVL_GRP = cur_grp
        self.N_QNS = len(QNS_PER_LV_GP[cur_grp])
        self.Q_ST = QNS_PER_LV_GP[cur_grp][0]
        self.Q_ED = QNS_PER_LV_GP[cur_grp][-1]

        # ==Mode==
        # Specify True to enable model training
        # self.train = True
        self.N_FOLD = 5

        # ==Data===
        # FEATS = ["et_diff", "event_comb_code", "room_fqid_code", "page_code", "text_fqid_code", "level_code"]
        # CAT_FEATS = ["event_comb_code", "room_fqid_code", "page_code","text_fqid_code", "level_code"]
        # COLS_TO_USE = ["session_id", "index", "level", "level_group", "elapsed_time", 
                        # "event_name", "name", "room_fqid", "page", "text_fqid"]

        self.NUM_FEATS = ["elapsed_time_log1p", "elapsed_time_diff_log1p", "room_coor_x", "room_coor_y"]
        self.CAT_FEATS = ['name', 'event_name', 'fqid', 'room_fqid', 'level']
        self.FEATS = self.NUM_FEATS + self.CAT_FEATS

        # T_WINDOW = 512
        self.T_WINDOW = 256
        
        # ==Training==
        self.SEED = 42
        # DEVICE = "cuda:0"
        self.DEVICE = "cpu"

        # EPOCH = 70
        self.EPOCH = 3
        self.CKPT_METRIC = "f1"
        # self.CKPT_METRIC = "f1@0.63"

        # ==DataLoader==
        # BATCH_SIZE = 128
        self.BATCH_SIZE = 3
        # NUM_WORKERS = 4
        self.NUM_WORKERS = 0

        # ==Solver==
        self.LR = 1e-3
        self.WEIGHT_DECAY = 1e-2

        # ==Early Stopping==
        # ES_PATIENCE = 20
        self.ES_PATIENCE = 5

        # ==Evaluator==
        self.EVAL_METRICS = ["auroc", "f1"]



def q2level(q):
    if q<=3: 
        grp = '0-4'
    elif q<=13: 
        grp = '5-12'
    elif q<=18: 
        grp = '13-22'

    return grp


event_name_feature = [
    # 'event_name_None',
    'cutscene_click', 'person_click', 'navigate_click', 'observation_click', 
    'notification_click', 'object_click', 'object_hover', 'map_hover', 
    'map_click', 'checkpoint', 'notebook_click'
]
name_feature = ['basic', 'undefined', 'close', 'open', 'prev', 'next']
fqid_feature = ['fqid_None', 'archivist', 'archivist_glasses', 'block', 'block_0', 'block_1', 'block_badge', 'block_badge_2', 'block_magnify', 'block_nelson', 'block_tocollection', 'block_tomap1', 'block_tomap2', 'boss', 'businesscards', 'businesscards.card_0.next', 'businesscards.card_1.next', 'businesscards.card_bingo.bingo', 'businesscards.card_bingo.next', 'ch3start', 'chap1_finale', 'chap1_finale_c', 'chap2_finale', 'chap2_finale_c', 'chap4_finale_c', 'coffee', 'colorbook', 'confrontation', 'crane_ranger', 'cs', 'directory', 'directory.closeup.archivist', 'door_block_clean', 'door_block_talk', 'doorblock', 'expert', 'flag_girl', 'fox', 'glasses', 'gramps', 'groupconvo', 'groupconvo_flag', 'intro', 'janitor', 'journals', 'journals.hub.topics', 'journals.pic_0.next', 'journals.pic_1.next', 'journals.pic_2.bingo', 'journals.pic_2.next', 'journals_flag', 'journals_flag.hub.topics', 'journals_flag.hub.topics_old', 'journals_flag.pic_0.bingo', 'journals_flag.pic_0.next', 'journals_flag.pic_0_old.next', 'journals_flag.pic_1.bingo', 'journals_flag.pic_1.next', 'journals_flag.pic_1_old.next', 'journals_flag.pic_2.bingo', 'journals_flag.pic_2.next', 'journals_flag.pic_2_old.next', 'key', 'lockeddoor', 'logbook', 'logbook.page.bingo', 'magnify', 'need_glasses', 'notebook', 'outtolunch', 'photo', 'plaque', 'plaque.face.date', 'reader', 'reader.paper0.next', 'reader.paper0.prev', 'reader.paper1.next', 'reader.paper1.prev', 'reader.paper2.bingo', 'reader.paper2.next', 'reader.paper2.prev', 'reader_flag', 'reader_flag.paper0.next', 'reader_flag.paper0.prev', 'reader_flag.paper1.next', 'reader_flag.paper1.prev', 'reader_flag.paper2.bingo', 'reader_flag.paper2.next', 'reader_flag.paper2.prev', 'remove_cup', 'report', 'retirement_letter', 'savedteddy', 'seescratches', 'teddy', 'tobasement', 'tocage', 'tocloset', 'tocloset_dirty', 'tocollection', 'tocollectionflag', 'toentry', 'tofrontdesk', 'togrampa', 'tohallway', 'tomap', 'tomicrofiche', 'tostacks', 'tracks', 'tracks.hub.deer', 'trigger_coffee', 'trigger_scarf', 'tunic', 'tunic.capitol_0', 'tunic.capitol_1', 'tunic.capitol_2', 'tunic.drycleaner', 'tunic.flaghouse', 'tunic.historicalsociety', 'tunic.hub.slip', 'tunic.humanecology', 'tunic.kohlcenter', 'tunic.library', 'tunic.wildlife', 'unlockdoor', 'wells', 'wellsbadge', 'what_happened', 'worker']
room_fqid_feature = [
    # 'room_fqid_None',
    'tunic.capitol_0.hall', 'tunic.capitol_1.hall', 'tunic.capitol_2.hall', 
    'tunic.drycleaner.frontdesk', 'tunic.flaghouse.entry', 
    'tunic.historicalsociety.basement', 'tunic.historicalsociety.cage','tunic.historicalsociety.closet', 
    'tunic.historicalsociety.closet_dirty', 'tunic.historicalsociety.collection', 'tunic.historicalsociety.collection_flag', 
    'tunic.historicalsociety.entry', 'tunic.historicalsociety.frontdesk', 'tunic.historicalsociety.stacks', 
    'tunic.humanecology.frontdesk', 'tunic.kohlcenter.halloffame', 
    'tunic.library.frontdesk', 'tunic.library.microfiche', 
    'tunic.wildlife.center'
]


name2int = {v:i for i, v in enumerate(name_feature)}
event2int = {v:i for i, v in enumerate(event_name_feature)}
fqid2int = {v:i for i, v in enumerate(fqid_feature)}
room2int = {v:i for i, v in enumerate(room_fqid_feature)}

print('cat len:', len(name_feature), len(event_name_feature), len(fqid_feature), len(room_fqid_feature))
# 6, 11, 129, 19
