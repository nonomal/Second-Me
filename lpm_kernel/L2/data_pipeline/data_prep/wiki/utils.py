import re
import random
import logging
import numpy as np
import Levenshtein
from enum import Enum
from datetime import datetime
from typing import List, Optional, Union, Any, Dict
from sklearn.metrics import pairwise_distances
from sklearn.cluster import AgglomerativeClustering


TIMESTAMP_PATTERN = r"\[_TIMESTAMP_\]\(.+\)"
TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
DATE_TIME_FORMAT = "%Y-%m-%d"
MONTH_TIME_FORMAT = "%Y-%m"


class MemoryType(str, Enum):
    TEXT = "TEXT"
    IMAGE = "IMAGE"
    AUDIO = "AUDIO"
    SHORT_AUDIO = "SHORT_AUDIO"
    PAGE = "PAGE"
    LINK="LINK"
    DOC="DOC"


class EntityType(str, Enum):
    PERSON = "PERSON"
    LOCATION = "LOCATION"
    NORMAL_ENTITY = "NORMAL_ENTITY"
    CONCEPT = "CONCEPT"
    USER_ATTRIBUTE = "USER_ATTRIBUTE"
    
    
class TimelineType(str, Enum):
    PERSON = "PERSON"
    ORGANIZATION = "ORGANIZATION"
    GEO = "GEO"
    PROPER_NOUN = "PROPER NOUN"
    COMMON_NOUN = "COMMON_NOUN"
    CONCEPT = "CONCEPT"

    @classmethod
    def _missing_(cls, value): 
        # default return SPECIFIC_OBJECT
        return cls.PROPER_NOUN
    

# TEMPLATE_MAP = {
#     MemoryType.IMAGE: IMAGE_TEMPLATES,
#     MemoryType.AUDIO: AUDIO_TEMPLATES,
#     MemoryType.TEXT: GENERAL_TEMPLATES,
#     MemoryType.SHORT_AUDIO: GENERAL_TEMPLATES,
#     MemoryType.PAGE: GENERAL_TEMPLATES
# }


class Note:
    def __init__(self,
                 noteId: int,
                 createTime: str,
                 memoryType: str,
                 userName: str,
                 insight: str="",
                 summary: str="",
                 content: str="",
                 title: str="",):
        self.id = noteId
        self.user_name = userName
        self.insight = re.sub(TIMESTAMP_PATTERN, '', insight) if insight else ""
        self.summary = summary
        self.content = content
        self.title = title
        self.create_time = createTime
        self.memory_type = MemoryType(memoryType)
        self.processed_content = self.get_processed_content(userName)
        
    
    def get_processed_content(self, user_name: str) -> str:
        if self.insight and self.content:
            return f"{self.insight}\n\n{self.content}"
        elif self.insight:
            return self.insight
        return self.content
    
    def to_str(self):
        return f"Note Id: {self.id}\n" + (f"Title: {self.title}\n" if self.title else "") + "\n" + self.processed_content

class Conversation:
    def __init__(self,
                 userName: str,
                 sessionId: str,
                 title: str,
                 summary: str,
                 createTime: str):
        self.user_name = userName
        self.session_id = sessionId
        self.title = title
        self.summary = summary
        self.create_time = createTime





class Timeline:
    """
    与后面的Monthly Timline可能会存在混淆歧义，简单来说这个timeline就是从单个memory中抽取得到的raw timeline
    由于monthly timeline需要按照创建时间进行归并，但是后端传进来的create time是raw timeline的创建时间，而不是note的创建时间，所以特意添加一个note_create_time字段
    """
    
    def __init__(self,
                 createTime: str,
                 noteId: int,
                 timelineType: Optional[Union[str, TimelineType]],
                 description: str,
                 content: str = "",  # 添加默认值
                 isUsed: bool=False,
                 noteCreateTime: str="",):
        self.create_time = createTime
        self.note_id = noteId
        self.description = description
        self.is_used = isUsed
        self.content = content  # 添加content字段
        
        if noteCreateTime:
            self.note_create_time = noteCreateTime
        else:
            self.note_create_time = createTime
        
        if isinstance(timelineType, str):
            self.timeline_type = TimelineType(timelineType)
        else:
            self.timeline_type = timelineType
    
    def _desc_(self, with_note_id: bool=False):
        if with_note_id:
            return f"[{time_format_shift(self.create_time)}][{self.note_id}] {self.description}"
        return f"[{time_format_shift(self.create_time)}] {self.description}"
    
    def _desc_4note(self, with_note_id: bool=False):
        if with_note_id:
            return f"[{time_format_shift(self.note_create_time)}][{self.note_id}] {self.description}"
        return f"[{time_format_shift(self.note_create_time)}] {self.description}"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "createTime": self.create_time,
            "noteId": self.note_id,
            "timelineType": self.timeline_type.value,
            "content": self.content,
            "description": self.description
        }


class Entity:
    def __init__(self,
                name: str,
                entityType: Optional[Union[str, EntityType]],
                synonyms: List[str],
                freq: int=0,
                genWiki: bool=False,
                description: str="",
                 **args):
        self.name = name
        self.freq = freq
        self.synonyms = synonyms
        self.gen_wiki = genWiki
        self.description = description
        self._args = args

        # 初始化评分相关的属性
        self.score = 0.0
        self.personal_uniqueness_reason = ""
        self.contextual_relevance_reason = ""
        self.personal_significance_reason = ""
        self.rarity_reason = ""
        self.time_relevance_reason = ""
        self.frequency_reason = ""
        self.emotional_connection_reason = ""

        if isinstance(entityType, str):
            self.entity_type = EntityType(entityType)
        else:
            self.entity_type = entityType

        if "timelines" in args:
            self.timelines = [Timeline(**timeline) if isinstance(timeline, dict) else timeline 
                              for timeline in args["timelines"]]
        else:
            self.timelines = []
            
    def merge_entity(self, entity: "Entity") -> None:
        self.freq += entity.freq
        self.synonyms = list(set(self.synonyms + entity.synonyms))
        self.timelines.extend(entity.timelines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "freq": self.freq,
            "entityType": self.entity_type.value,
            "genWiki": self.gen_wiki,
            "synonyms": self.synonyms,
            "timelines": [timeline.to_dict() for timeline in self.timelines],
            "score": self.score,
            "personal_uniqueness_reason": self.personal_uniqueness_reason,
            "contextual_relevance_reason": self.contextual_relevance_reason,
            "personal_significance_reason": self.personal_significance_reason,
            "rarity_reason": self.rarity_reason,
            "time_relevance_reason": self.time_relevance_reason,
            "frequency_reason": self.frequency_reason,
            "emotional_connection_reason": self.emotional_connection_reason
        }

    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        elif key in self._args:
            return self._args[key]
        else:
            raise KeyError(f"Key '{key}' not found in Entity.")

    def __setitem__(self, key: str, value: Any) -> None:
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            self._args[key] = value


class Concept(Entity):
    def __init__(self,
                 id=None,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.id = id
                 
    def improve_timelines(self, timelines: List[Timeline]) -> None:
        self.timelines.extend(timelines)
        self.freq += len(timelines)           
    
    def merge_concept(self, concept: "Concept") -> None:
        self.freq += concept.freq
        self.synonyms = list(set(self.synonyms + concept.synonyms))
        self.timelines.extend(concept.timelines)
                 
    def to_str(self):
        return f"Name: {self.name}\nDescription: {self.description}\n\n"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            **super().to_dict()
        }


class DailyTimeline:
    def __init__(self,
                 id: int,
                 dateTime: str,
                 content: str,
                 noteIds: List[int]):
        self.id = id
        self.date_time = dateTime
        self.content = content.strip()
        self.note_ids = noteIds


    def _desc_(self):
        return f"- [{self.date_time}] {self.content}".strip()
    
    def to_dict(self):
        return {
            "id": self.id,
            "dateTime": self.date_time,
            "content": self.content,
            "noteIds": self.note_ids
        }


class MonthlyTimeline:
    def __init__(self,
                 id: int,
                 monthDate: str,
                 title: str,
                 dailyTimelines: List[Dict[str, Any]]):
        self.id = id
        self.month_date = monthDate
        self.title = title
        daily_timelines = [DailyTimeline(**daily_timeline) for daily_timeline in dailyTimelines]
        self.daily_timelines = sorted(daily_timelines, key=lambda x: datetime.strptime(x.date_time, DATE_TIME_FORMAT))
        
    def _desc_(self):
        return f"** {self.month_date} **\n" + "\n".join([daily_timeline._desc_() for daily_timeline in self.daily_timelines])
        
    def _preview_(self, preview_num: int=0):
        preview_statement = f"[{self.month_date}] {self.title}\n"
        for daily_timeline in self.daily_timelines[:preview_num]:
            preview_statement += daily_timeline._desc_() + "\n"
        return preview_statement
    
    def to_dict(self):
        return {
            "id": self.id,
            "monthDate": self.month_date,
            "title": self.title,
            "dailyTimelines": [daily_timeline.to_dict() for daily_timeline in self.daily_timelines]
        }


class EntityWiki:
    def __init__(self,
                 wikiText: str,
                 monthlyTimelines: List[Dict[str, Any]]=[]):
        self.wiki_text = wikiText
        self.monthly_timelines=[]
        if monthlyTimelines:
            self.monthly_timelines = [MonthlyTimeline(**monthly_timeline) for monthly_timeline in monthlyTimelines]
        self.max_month_idx = max([monthly_timeline.id for monthly_timeline in self.monthly_timelines]) if self.monthly_timelines else 0

    def to_dict(self):
        return {
            "wikiText": self.wiki_text,
            "monthlyTimelines": [monthly_timeline.to_dict() for monthly_timeline in self.monthly_timelines]
        }


def levenshtein_distance(s1: str, s2: str) -> int:
    return Levenshtein.distance(s1, s2)

        
def build_clusters(entity_names: List[str], distance_threshold: int) -> List[List[str]]:
    if len(entity_names) < 2:
        return [entity_names]
    
    entity_names_array = np.array(entity_names)
    distance_matrix = pairwise_distances(entity_names_array, metric=lambda x, y: levenshtein_distance(x, y))
    clustering = AgglomerativeClustering(distance_threshold=distance_threshold, 
                                         n_clusters=None, 
                                         metric="precomputed", 
                                         linkage="average")
    clustering.fit(distance_matrix)
    labels = clustering.labels_
    
    clusters = {}
    for label, entity_name in zip(labels, entity_names):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(entity_name)
    return list(clusters.values())


def find_neibor_entities(center_entities: List[Entity], entities: List[Entity], neibor_num: int=5) -> Dict[str, List[str]]:
    center_entity_names = np.array([entity.name for entity in center_entities])
    entity_names = np.array([entity.name for entity in entities])
    
    center_entity_vec_list = [np.array([es.lower() for es in entity.synonyms]) for entity in center_entities]
    entity_candidates_vec_list = [np.array([s.lower() for s in  entity.synonyms]) for entity in entities]
    # shape in [new x old] matrix
    distance_matrix = np.array([
        [
            np.min(pairwise_distances(center_entity_vec, entity_candidates_vec, metric=lambda x, y: levenshtein_distance(x, y)))
            for entity_candidates_vec in entity_candidates_vec_list
        ]
        for center_entity_vec in center_entity_vec_list
    ])
    
    neibor_indices = np.argsort(distance_matrix, axis=1)[:, :neibor_num]
    neibor_entity_names = entity_names[neibor_indices]

    return {
        center_entity_name: neibor_entity_names[i].tolist() 
        for i, center_entity_name in enumerate(center_entity_names)
    }


def parse_monthly_timelines(monthly_timeline_str: str, month_date: str, month_idx: int=0) -> List[MonthlyTimeline]:
    day_idx = 0
    monthly_timeline_dict = {}
    month_pattern = r"\*\*\s*(\d{4}-\d{2})\s*\*\*"
    timeline_pattern = r"-\s*\[(\d{4}-\d{2}-\d{2})\]\s*(.*?)\[(.*?)\]$"
    current_month = None

    # 打印处理前的 monthly_timeline_str
    logging.info(f"Before processing:\n{monthly_timeline_str}")
    
    for line in monthly_timeline_str.split('\n'):
        month_match = re.match(month_pattern, line.strip())
        if month_match:
            current_month = month_match.group(1)
            monthly_timeline_dict[current_month] = {
                "id": month_idx,
                "monthDate": current_month, 
                "title": "",
                "dailyTimelines": []
            }
            month_idx += 1
        else:
            timeline_match = re.match(timeline_pattern, line.strip())
            if timeline_match and current_month is not None:
                date = timeline_match.group(1)
                content = timeline_match.group(2).strip()
                ## Compatible noteIds format of [1](\s[2])* and [1(,\s2)*]
                noteIds = timeline_match.group(3).replace(" ", "").replace("][", ",").split(",")
                month = time_format_shift(date, month_format=True)
                daily_timeline = {
                    "id": day_idx,
                    "dateTime": date,
                    "content": content,
                    "noteIds": [int(noteId) for noteId in noteIds]
                }
                if month not in monthly_timeline_dict:
                    monthly_timeline_dict[month] = {
                        "id": month_idx,
                        "monthDate": month,
                        "title": "",
                        "dailyTimelines": []
                    }
                monthly_timeline_dict[month]["dailyTimelines"].append(daily_timeline)
                day_idx += 1
    
    logging.info(f"After processing:\n{monthly_timeline_dict}")

    # 检查 monthly_timeline_dict 中是否存在 month_date 键
    if month_date in monthly_timeline_dict:
        month_timelines = MonthlyTimeline(**monthly_timeline_dict[month_date])
    else:
        # 创建一个空的 MonthlyTimeline 对象
        month_timelines = MonthlyTimeline(id=month_idx, monthDate=month_date, title="", dailyTimelines=[])
    
    return month_timelines
def parse_daily_timelines(timeline_str: str, day_date: str) -> Dict[str, Any]:
    daily_timeline_dict = {}
    timeline_pattern = r"-\s*\[(\d{4}-\d{2}-\d{2})\]\s*(.*?)\[(.*?)\]$"
    
    # 打印处理前的 timeline_str
    print(f"Before processing:\n{timeline_str}")
    
    for line in timeline_str.split('\n'):
        timeline_match = re.match(timeline_pattern, line.strip())
        if timeline_match:
            date = timeline_match.group(1)
            content = timeline_match.group(2).strip()
            ## Compatible noteIds format of [1](\s[2])* and [1(,\s2)*]
            noteIds = timeline_match.group(3).replace(" ", "").replace("][", ",").split(",")
            if date == day_date:
                daily_timeline_dict = {
                    "id": 0,  # Placeholder, will be set by the caller
                    "dateTime": date,
                    "content": content,
                    "noteIds": [int(noteId) for noteId in noteIds]
                }
                break
    
    print(f"After processing:\n{daily_timeline_dict}")

    return daily_timeline_dict if daily_timeline_dict else {"id": 0, "dateTime": "", "content": "", "noteIds": []}
    

def group_timelines_by_month(timelines: List[Timeline]) -> Dict[str, List[Timeline]]:
    monthly_timelines = {}
    for timeline in timelines:
        # 由于raw timeline和note的创建时间存在gap，所以选择使用note create time
        month = datetime.strptime(timeline.note_create_time, TIME_FORMAT).strftime(MONTH_TIME_FORMAT)
        if month not in monthly_timelines:
            monthly_timelines[month] = []
        monthly_timelines[month].append(timeline)
    return monthly_timelines

def group_timelines_by_day(timelines: List[Timeline]) -> Dict[str, List[Timeline]]:
    daily_timelines = {}
    for timeline in timelines:
        # 使用note的创建时间进行归并
        day = datetime.strptime(timeline.note_create_time, TIME_FORMAT).strftime(DATE_TIME_FORMAT)
        if day not in daily_timelines:
            daily_timelines[day] = []
        daily_timelines[day].append(timeline)
    return daily_timelines



def datetime2timestamp(time_str: str) -> float:
    try:
        timestamp = datetime.strptime(time_str, TIME_FORMAT).timestamp()
        return timestamp
    except Exception as e:
        logging.error(f"Invalid time format: {time_str}")
        raise e
    

# def time_format_shift(time_str: str, month_format: bool=False) -> datetime:
#     try:
#         if month_format:
#             return datetime.strptime(time_str, DATE_TIME_FORMAT).strftime(MONTH_TIME_FORMAT)
#         return datetime.strptime(time_str, TIME_FORMAT).strftime(DATE_TIME_FORMAT)
#     except Exception as e:
#         print(f"Invalid time format: {time_str}")
#         raise e
def time_format_shift(time_str: str, month_format: bool=False) -> str:
    # 处理空字符串情况
    if not time_str:
        return "未知时间"
    
    try:
        if month_format:
            return datetime.strptime(time_str, DATE_TIME_FORMAT).strftime(MONTH_TIME_FORMAT)
        return datetime.strptime(time_str, TIME_FORMAT).strftime(DATE_TIME_FORMAT)
    except Exception as e:
        logging.error(f"Invalid time format: {time_str}")
        # 出错时返回一个默认值而不是抛出异常
        return "未知时间"
    

def is_valid_note(note: Dict[str, Any]) -> bool:
    if not (note.get("insight", "") or note.get("summary", "") or note.get("content", "")):
        return False
    return True
