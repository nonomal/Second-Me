import logging
from typing import List, Optional, Dict, Any, Union
from datetime import datetime

from lpm_kernel.L2.data_pipeline.data_prep.wiki.utils import Note, Conversation

ENTITY_TYPES = ["PERSON", "GEO", "ORGANIZATION", "PROPER NOUN", "COMMON NOUN", "USER ATTRIBUTE"]
TUPLE_DELIMITER = "<|>"
RECORD_DELIMITER = "##"
START_DELIMITER = "<|START|>"
COMPLETION_DELIMITER = "<|COMPLETE|>"
TIME_FORMAT = "%Y-%m-%d %H:%M:%S"


class Prompts:
    
    PREFER_LANGUAGE_SYSTEM_PROMPT = """User preferred to use {language} language, you should use the language in the appropriate fields during the generation process, but retain the original language for some special proper nouns."""
    
    ENTITY_EXTRACT_SYSTEM_PROMPT = """# **Task Overview**:
To build a personal encyclopedia named "Personal Wiki" exclusively for {userName}, extracting meaningful entities from daily notes and conversations. 
Focus on identifying personalized nouns that: 
1. Nouns that recur in user-generated content; 
2. Hold unique personal significance compared to general population; 
3. Carry user-specific memories or emotional connections. Strictly filter out generic entities lacking personalization attributes, with an enhanced emphasis on extracting names of people related to {userName} (e.g., friends, colleagues, family members, etc.), professional terms, high-frequency words, etc.

## 1.Perspective Setting:
* Always describe entities from {userName}'s perspective, using {userName} as the third-person subject in all descriptions.
* Clearly distinguish between {userName}'s perspective and other users' experiences when mentioned.

## 2.Entity Analysis:
* Analyze the text thoroughly, focusing on explicit and implicit references to people, objects, and events.
* Highlight how these entities influence {userName}'s worldview, beliefs, professional activities, or personal life.
* Include only entities that directly impact, challenge, or align with {userName}'s core values, experiences, or goals.

## 3.Extraction Strategy:
* Extract entities highly relevant to {userName}, considering both professional and daily-life contexts.
* Avoid generic or irrelevant entities unless they are crucial to understanding {userName}'s cognitive or emotional changes.
* Only extract entities contained in the original notes, avoiding summarizing new nouns.Focus on physical nouns and people rather than abstract concepts.

## 4.Entity Structure:

Each entity should include:
* Entity Name: The name of the entity as it appears in the record.
* Entity Type: The type of the entity (from {entity_types}).
* Entity Description: Capture {userName}'s impression of the entity in {prefer_lang}, emphasizing its relevance to {userName}'s background and experiences.
* Create Time: The timestamp of the note or conversation where the entity appears.
* Note Id: The Note Id of the note supplied by the origin information, if there's no note Id, then it should be ''.
* Session Id: The Session Id of the conversation supplied by the origin information, if there's no session Id, then it should be ''.

## 5.Output Specifications:
* Use {prefer_lang} for all descriptions but do not translate {userName}'s name.
* Ensure clarity in describing each entity, maintaining a professional yet accessible tone, and avoid overly complicated or casual expressions.
* For each extracted entity, show the note id or session id and the create time of the origin note it was extracted from, so follow-up processes can clearly find the relationship with the entity and the note or conversations. For entities extracted from multiple notes, list all notes containing the entity.
* Focus solely on entities relevant to {userName}, avoiding generic or unrelated ones unless their inclusion is critical to understanding {userName}'s emotional or cognitive transformation.

## 6.Special Notes: 
* Exclude entities that lack meaningful connections to {userName}'s background or experiences.
* Contextual relevance is key; avoid including frequently mentioned entities if they have no tangible link to {userName}'s personal or professional life.

# **Output Specifications**
Your output should be in json format{{[\"entity_name\":xxx, \"entity_description\":xxx, \"entity_type\": xxx, \"notes\":[{{\"note_id\", \"create_time\"}}]]}}
"""
    ENTITY_EXTRACT_SYSTEM_PROMPT_simplfied = """
You are expert on extracting entity from the provided text, using only entity names from the input record and generating descriptions in {prefer_lang} from {userName}'s perspective.

## Extraction Strategy:
You should extract entities highly relevant to {userName}, considering both professional and daily-life contexts. Additionally, only extract entities contained in the original notes, avoiding summarizing new nouns.Focus on physical nouns and people rather than abstract concepts.

## Entity Structure:
For the entity structure, Each entity should include:
* Entity Name: The name of the entity as it appears in the record.
* Entity Type: The type of the entity (from {entity_types}).
* Entity Description: Capture {userName}'s impression of the entity in {prefer_lang}, emphasizing its relevance to {userName}'s background and experiences.
* Create Time: The timestamp of the note or conversation where the entity appears.
* Note Id: The Note Id of the note supplied by the origin information, if there's no note Id, then it should be ''.
* Session Id: The Session Id of the conversation supplied by the origin information, if there's no session Id, then it should be ''.

## Output Specifications:
* Use {prefer_lang} for all descriptions but do not translate {userName}'s name.
* Ensure clarity in describing each entity, maintaining a professional yet accessible tone, and avoid overly complicated or casual expressions.
* For each extracted entity, show the note id or session id and the create time of the origin note it was extracted from, so follow-up processes can clearly find the relationship with the entity and the note or conversations. 
* You should extract at least 2 entities from one notes.
* Your output should be in json format{{[\"entity_name\":xxx, \"entity_description\":xxx, \"entity_type\": xxx, \"notes\":[{{\"note_id\", \"create_time\"}}]]}}
    """

    ENTITY_EXTRACT_USER_PROMPT = """
### Record ### [Extract entities from this part]
{user_text}
"""

    ENTITY_EXTRACT_SYSTEM_PROMPT_gt_v1 = """
Your task is to extract entities and their descriptions from the provided text, using only entity names from the input record and generating descriptions in {prefer_lang} from {userName}'s perspective.

1. Perspective Setting:

    - Always use {userName} as the third-person subject in descriptions.

    - All entity impressions should be described from {userName}'s perspective.

    - When notes mention other users' experiences, clearly distinguish them from {userName}'s perspective.

2. User Profile Analysis:

    - Construct user persona from self-introduction, focusing on key characteristics, interests, professional/personal background, and potential value system and preferences (only if self-introduction is available).

3. Record Content Analysis:

    - Deeply comprehend the record content, focusing on core events, {userName}'s cognitive changes, emotional responses, and relationships between entities.

4. Entity Extraction and Description:

    Entity Structure:

        - Entity Name: Name of the entity (from record only).

        - Entity Type: Type of the entity (from {entity_types}).

        - Entity Desc: Capture {userName}'s impression of the entity in {prefer_lang}, balancing subjective impressions with factual context.

5. Personalization and Expression Strategy:

    - Use {prefer_lang} for all descriptions, maintaining a natural, concise, and consistent third-person perspective using {userName}.

    - Choose expressions based on {userName}'s background, using a clear and structured narrative flow with subtle emotional undertones.

6. Output Guidelines:

    - Lead with key observations, support with relevant details, and include perception changes when present.

    - Ensure descriptions are clear, natural, and in {prefer_lang}, maintaining a professional yet accessible tone, and keep entity names mentioned in the description remain as in the original record.

    - Avoid excessive emotional expression, complex structures, mixed perspectives, and overly casual language.

Special Notes:

Do not extract any entities from the few-shot examples provided. Focus solely on entities from the actual input and ensure all descriptions are in {prefer_lang}.

Few-Shot Examples:

Input:
userName: "小明"
User Introduction: "从事金融行业的数据分析师，平时工作压力较大，喜欢通过阅读和运动放松。对投资理财话题特别关注。"
Record: "今天读完《穷查理宝典》，芒格的思维方式真的很特别。一开始觉得这本书可能很枯燥，但读完发现里面讲的多元思维模型特别实用，尤其是在分析投资决策时。最近工作上的一些分析思路也受到了启发。"
Output:
{start_delimiter}
### Entity Extraction
1. **Entity Name**: 《穷查理宝典》
   **Entity Type**: normal_entity
   **Entity Desc**: 这本书是小明认知转变的重要节点。小明从最初认为内容可能枯燥的预设，转变为深度认可书中的思维方法。特别是书中的多元思维模型引起了小明的专业共鸣，对其工作分析方法产生了积极影响。

2. **Entity Name**: 芒格
   **Entity Type**: person
   **Entity Desc**: 对小明产生了显著的思维影响。小明通过了解其思维方式，在投资决策和数据分析领域获得了新的视角，与小明的职业背景和兴趣高度相关。
{completion_delimiter}
"""
    ENTITY_FILTER_USER_PROMPT = """
Context Information:
1. Entity to Evaluate: {entity_name}
2. User Background: {global_bio}
3. Entity Statistics:
   - Total Appearances: {entity_freq}
   - First Appearance: {first_appearance}
   - Last Appearance: {last_appearance}
   - Time Span: {time_span} days
   - Appearance Pattern: {appearance_pattern}
4. Current Time: {cur_time}

Notes containing the entity (chronologically ordered):
{entity_notes_text}

Please evaluate this entity's significance based on the scoring dimensions outlined above.
"""

    ENTITY_EXTRACT_FILTER_SYSTEM_PROMPT = """
Task Overview:
Evaluate the significance of each entity extracted from given notes using seven dimensions. Each dimension is scored on a 0-2 scale.

Scoring Dimensions:

1. Personal Uniqueness
Score 2:
- Entity is deeply personal to the user (e.g., family members, close friends, personal projects)
- Represents unique experiences or achievements
- Has direct impact on user's personal life
Example: "My thesis project", "My mentor Dr. Zhang"

Score 1:
- Entity has some personal connection but is shared with others
- Moderate level of personal involvement
Example: "Company team project", "Department colleague"

Score 0:
- Generic entity with no personal connection
- Common knowledge or public information
Example: "General market trends", "Public figures without direct interaction"

2. Contextual Relevance
Score 2:
- Entity is central to understanding multiple user experiences
- Appears in critical contexts or decision points
- Connects multiple aspects of user's life
Example: "Programming language used in both work and personal projects", "Mentor who influences career and personal growth"

Score 1:
- Entity provides useful context but isn't central
- Appears in limited contexts
Example: "Tool used occasionally at work", "Conference attended once"

Score 0:
- Entity appears in isolation
- Lacks connection to other experiences
Example: "Random news article", "One-time encounter"

3. Personal Significance
Score 2:
- Entity has profound impact on user's life
- Influences major decisions or life changes
- Strong emotional connection
Example: "Career-changing project", "Life-altering event"

Score 1:
- Moderate impact on user's life
- Some influence on decisions
Example: "Useful work tool", "Helpful colleague"

Score 0:
- Minimal impact on user's life
- No influence on decisions
Example: "Casual acquaintance", "Minor event"

4. Rarity
Score 2:
- Highly specialized or unique entity
- Rare occurrence in general context
Example: "Cutting-edge technology", "Unique research finding"

Score 1:
- Moderately uncommon
- Specialized but not rare
Example: "Industry-specific tool", "Professional certification"

Score 0:
- Common or everyday entity
Example: "Common software", "Basic concept"

5. Time Relevance
Score 2:
- Currently active or highly relevant
- Mentioned within last week
- Ongoing importance

Score 1:
- Recent but not current
- Mentioned within last month

Score 0:
- Historical or outdated
- Mentioned over a month ago

6. Frequency
Score 2:
- Mentioned > 6 times
- Consistent presence across notes

Score 1:
- Mentioned 3-6 times
- Regular but not frequent

Score 0:
- Mentioned < 3 times
- Sporadic appearance

7. Emotional Connection
Score 2:
- Strong emotional response or attachment
- Deep personal investment
Example: "Dream project", "Close mentor"

Score 1:
- Moderate emotional connection
- Professional interest
Example: "Interesting task", "Helpful resource"

Score 0:
- No emotional connection
- Purely factual relationship
Example: "Basic tool", "Standard procedure"

Final Score Calculation:
1. Each dimension score (0-2) is evaluated to determine significance.
2. For generating a wiki:
   - Generate wiki if overall impression is highly significant.
   - Consider generating wiki with moderate significance if frequency is high.

Output Format:
You must respond with a JSON object in the following format:

```json
{
    "personal_uniqueness": {
        "reason": "Detailed explanation with specific examples from notes",
        "score": 0-2
    },
    "contextual_relevance": {
        "reason": "Detailed explanation with specific examples from notes",
        "score": 0-2
    },
    "personal_significance": {
        "reason": "Detailed explanation with specific examples from notes",
        "score": 0-2
    },
    "rarity": {
        "reason": "Detailed explanation with specific examples from notes",
        "score": 0-2
    },
    "time_relevance": {
        "reason": "Detailed explanation with specific examples from notes",
        "score": 0-2
    },
    "frequency": {
        "reason": "Detailed explanation with specific examples from notes",
        "score": 0-2
    },
    "emotional_connection": {
        "reason": "Detailed explanation with specific examples from notes",
        "score": 0-2
    },
    "weighted_final_score": 0-2
}
```

Your response must be valid JSON and include all fields exactly as shown above. The score values must be numbers between 0 and 2.
"""
    ENTITY_EXTRACT_SYSTEM_PROMPT_batch_test = """
Your name is {userName}, this is how others see you: {global_bio}.
As I know you are a deep note-taker, so now I have one thing that want to cooperate with you.
Now I want to create a personal wiki for you, as you see it is personal, meaning it is an encyclopaedia for you only. It's your own encyclopedia, unlike wikipedia, which records what everyone knows, so the entries will be nouns that appeared in your notes and for every one of them, it have special meaning for you.
So I want you to help me to find out the entries of the wiki from your notes. Remember, the wiki is for yourself, not for everyone. And also, a hope your work can follow these guidance:
1. Perspective Setting:
Always describe entities from {userName}'s perspective, using {userName} as the third-person subject in all descriptions.
Distinguish clearly between {userName}'s perspective and other users' experiences when mentioned.
2. Record Content Analysis (Focus on Entities):
Analyze the record in depth, concentrating on explicit and implicit references to people and objects. Pay particular attention to how {userName} interacts with, perceives, and is influenced by these entities. Track {userName}'s cognitive responses (e.g., changes in thinking or perspective) and emotional shifts, especially as they relate to specific people, relationships, and physical objects.
Focus on how these entities shape {userName}'s worldview, beliefs, or professional activities. Highlight relationships with significant individuals and meaningful objects that play a role in the user's personal or professional life. Only include entities that directly impact, challenge, or align with {userName}'s core values, experiences, or goals.
**Output Specifications**

{start_delimiter}
### Entity Extraction

1. **Entity Name**: xxx
   **Entity Type**: xxx
   **Entity Desc**: xxx
   **Create Time**: xxx
   **Note Id**: xxx
   **Session Id**: xxx
2. xxx
3. xxx
...
    """
    ENTITY_EXTRACT_SYSTEM_PROMPT_batch_v2 = """
Background:
I'm creating a tool named personal wiki, it is an encyclopaedia for the user only, summarized from the user's daily notes and conversations. So the entries in this wiki should be nouns that appeared in user's notes or conversations, and they should be very PERSONAL to the user, means that they should have special meaning for the user comparing with other people.
Task Overview:
Your task is to extract RELEVANT entities and their descriptions from the provided text, focusing on entities that are closely linked to the personal background and the context of the text. Do not extract generic entities that do not resonate with {userName}'s experiences. 

1. Perspective Setting:
Always describe entities from {userName}'s perspective, using {userName} as the third-person subject in all descriptions.
Distinguish clearly between {userName}'s perspective and other users' experiences when mentioned.
2. Record Content Analysis (Focus on Entities):
Analyze the record in depth, concentrating on explicit and implicit references to people and objects. Pay particular attention to how {userName} interacts with, and is influenced by these entities. Try to get the facts straight, focus more on the relationship and experience instead of making emotional assumptions.
Focus on how these entities shape {userName}'s worldview, beliefs, or professional activities. Highlight relationships with significant individuals and meaningful objects that play a role in the user's personal or professional life. Only include entities that directly impact, challenge, or align with {userName}'s core values, experiences, or goals.
3. Entity Extraction Strategy:
Only extract entities that are highly relevant to {userName}, considering both their professional background and their daily life. Avoid generic or irrelevant entities unless they are crucial to understanding the user's cognitive or emotional changes.
Only entities that are directly tied to {userName}'s professional life, daily life, interpersonal relationship  should be extracted. Only the entities contained in the origin notes should be extracted, DO NOT try to summary new nouns by yourself. Focus on physical nouns and people rather than conceptual or abstract ones. 
4. Entity Extraction and Description:
Entity Structure:
Entity Name: The name of the entity (as it appears in the record).
Entity Type: The type of the entity (from {entity_types}).
Entity Description: Capture {userName}'s impression of the entity in {prefer_lang}, while emphasizing its relevance to {userName}'s background and experiences.
Create time: The create time of the note supplied by the origin information.
Note Id: The Note Id of the note supplied by the origin information, if there's no note Id, only a session Id exists, then the note Id should be ''
Session Id: The Session Id of the conversation supplied by the origin information, if there's no session Id, only a note Id, then the session Id should be ''
If the entity influences {userName}'s thinking or work (e.g., an author, concept, or person), highlight how this entity directly impacts {userName}'s professional, personal, or intellectual life.
Avoid including descriptions of entities that are generic or unrelated to the user unless they directly help clarify something important about {userName}'s experience.
5. Output Guidelines:
Use {prefer_lang} for all descriptions, ensuring that {userName}'s perspective remains consistent in the third person, with natural, concise, and clear expressions, don't simply use his/her instead.
Begin with key observations from {userName}'s perspective, followed by relevant details that explain the connection between the entity and {userName}'s professional/personal experiences.
Ensure clarity in how each entity is described, maintaining a professional yet accessible tone, and avoid overly complicated or casual expressions.
Focus solely on entities that are relevant to {userName}, and avoid extracting generic or unrelated ones unless their inclusion is critical to understanding {userName}'s emotional or cognitive transformation.
Special Notes:
Do not extract entities that do not meaningfully relate to {userName}'s background, or experiences. This means entities like generic concepts or figures that do not connect with {userName}'s profession or daily life should be excluded.
Contextual relevance is key: Even if an entity appears frequently in the record, if it has no tangible link to {userName}'s personal or professional life, it should not be included.
Also, for each entity you extracted, you need to show the note id or session id and the create time of the origin note it extracted from, so that follow up processes can clearly find the relationship with the entity and the note or conversations. For entities that extracted from multiple notes, you should list all the notes which the entity appeared.
**Output Specifications**

{start_delimiter}
### Entity Extraction

1. **Entity Name**: xxx
   **Entity Type**: xxx
   **Entity Desc**: xxx
   **Create Time**: xxx
   **Note Id**: xxx
   **Session Id**: xxx
2. xxx
3. xxx
...
    """
    GENERATE_TIMELINE_BY_NOTE_SYSTEM_PROMPT = """
    Background:
I am creating a tool called 'Personal Wiki,' which serves as a personalized encyclopedia for the user, summarizing information from their daily notes and conversations. The entries in this wiki consist of nouns that appear in the user's notes or conversations, and they are highly personal to the user—meaning they have special significance compared to others. Now, I have generated the wiki entries and want to show the user their interactions with each entity, as well as the development and changes in their relationship with each one.
Task Overview:
Your task is to generate a timeline from a given note for the entity contained in this note, restore the facts in your notes as fully as possible.

1. Perspective Setting:
Always describe entities from {userName}'s perspective, using {userName} as the third-person subject in all descriptions.
Distinguish clearly between {userName}'s perspective and other users' experiences when mentioned.
2. Record Content Analysis
Focus on how the entities shape {userName}'s worldview, beliefs, or professional activities. Highlight relationships with significant individuals and meaningful objects that play a role in the user’s personal or professional life. If the entity influences {userName}'s thinking or work (e.g., an author, concept, or person), highlight how this entity directly impacts {userName}'s professional, personal, or intellectual life. Comparing with a summary, it's better to be a detailed description of the note. If there is no information about how the entity influences {userName}'s life, you should output the original content. Avoid overly interpretation.
3. Output Guidelines:
- Use {prefer_lang} for the whole descriptions, ensuring that {userName}'s perspective remains consistent in the third person, with natural, concise, and clear expressions, don't simply use his/her instead.
- Begin with key observations from {userName}'s perspective, followed by relevant details that explain the connection between the entity and {userName}'s professional/personal experiences.
- Ensure clarity in how the entity is described, maintaining a professional yet accessible tone, and avoid overly complicated or casual expressions.
Just generate the content, a title is not necessary.
DO NOT TRY TO TRANSLATE user's name.
- Keep your output concise, no more than two sentences. And you should respond in the following Output Specifications:

**Output Specifications**

{start_delimiter}
### Entity Timeline

**Timeline**: xxx
...
    """

    @staticmethod
    def return_entity_extract_prompt(prefer_lang: str,
                                     entity_input: List[Union[Note, Conversation]],
                                     user_name: str,
                                     user_self_intro: str,
                                     global_bio:str,
                                     system_prompt: str,
                                     need_few_shot: bool=True) -> str:
        format_dict = {
            "entity_types": ",".join(ENTITY_TYPES),
            "prefer_lang": prefer_lang,
            "start_delimiter": START_DELIMITER,
            "completion_delimiter": COMPLETION_DELIMITER,
            "userName":user_name,
            "global_bio": global_bio
        }
        
        if not system_prompt:
            system_prompt = Prompts.ENTITY_EXTRACT_SYSTEM_PROMPT_batch_v2

        
        system_messages = [
            {
                "role": "system",
                "content": system_prompt.format(**format_dict)
            }
        ]

        user_text = "\n\n"
        for entity in entity_input:
            if isinstance(entity, Note):
                user_text += f"Text_Type:Note\nnote_id:{entity.id}\ntext:{entity.processed_content}\ncreate_time:{entity.create_time}\n\n"
            else:
                user_text += f"Text_Type:conversation\nsession_id:{entity.session_id}\ntitle:{entity.title}\nsummary:{entity.summary}\ncreate_time:{entity.create_time}\n\n"

        user_messages = [
            {
                "role": "user",
                "content": Prompts.ENTITY_EXTRACT_USER_PROMPT.format(**format_dict,
                                                                     user_text=user_text)
            }
        ]
        
        if not need_few_shot:
            return system_messages + user_messages
        
        few_shots = []
        
        return system_messages + few_shots + user_messages

    @staticmethod
    def generate_timeline_by_notes(user_name, preferred_language, entity, origin_note, system_prompt) -> str:
        format_dict = {"userName":user_name, "prefer_lang": preferred_language, "start_delimiter": START_DELIMITER}
        entity_name = entity["name"]
        if not system_prompt:
            system_prompt = Prompts.GENERATE_TIMELINE_BY_NOTE_SYSTEM_PROMPT
        system_messages = [
            {
                "role": "system",
                "content": system_prompt.format(**format_dict)
            }
        ]
        user_text = f"user name is: {user_name}\n\n"
        user_text += f"The entity is: {entity_name}"
        user_text += f"Text_Type:Note\nnote_id:{origin_note.id}\ntext:{origin_note.processed_content}\ncreate_time:{origin_note.create_time}\n\n"
        user_messages = [
            {
                "role": "user",
                "content": user_text
            }
        ]
        return system_messages + user_messages

    @staticmethod
    def return_introspection_extracted_entities_prompt(entity_name: str,
                                                       user_name: str,
                                                       input_notes: List[Note],
                                                       global_bio: str,
                                                       system_prompt: str) -> str:
        if not system_prompt:
            system_prompt = Prompts.ENTITY_EXTRACT_FILTER_SYSTEM_PROMPT
            
        # 按时间排序笔记
        sorted_notes = sorted(input_notes, key=lambda x: datetime.strptime(x.create_time, TIME_FORMAT))
        
        # 计算实体统计信息
        entity_freq = len(sorted_notes)
        first_appearance = sorted_notes[0].create_time if sorted_notes else ""
        last_appearance = sorted_notes[-1].create_time if sorted_notes else ""
        
        # 计算时间跨度（天数）
        if first_appearance and last_appearance:
            first_date = datetime.strptime(first_appearance, TIME_FORMAT)
            last_date = datetime.strptime(last_appearance, TIME_FORMAT)
            time_span = (last_date - first_date).days
        else:
            time_span = 0
            
        # 分析出现模式
        if entity_freq <= 2:
            appearance_pattern = "Rare"
        elif time_span == 0:
            appearance_pattern = "Concentrated"
        elif entity_freq / (time_span + 1) >= 0.5:
            appearance_pattern = "Frequent"
        else:
            appearance_pattern = "Sporadic"
        
        # 构建结构化的笔记文本
        note_texts = []
        for note in sorted_notes:
            note_text = f"[Time: {note.create_time}]"
            if note.title:
                note_text += f"\nTitle: {note.title}"
            note_text += f"\nContent: {note.processed_content}\n"
            note_texts.append(note_text)
            
        entity_notes_text = "\n---\n".join(note_texts)
        
        format_dict = {
            "entity_name": entity_name,
            "userName": user_name,
            "global_bio": global_bio,
            "start_delimiter": START_DELIMITER,
            "entity_notes_text": entity_notes_text,
            "entity_freq": entity_freq,
            "first_appearance": first_appearance,
            "last_appearance": last_appearance,
            "time_span": time_span,
            "appearance_pattern": appearance_pattern,
            "cur_time": datetime.now().strftime(TIME_FORMAT)
        }
        
        system_messages = [
            {
                "role": "system",
                "content": system_prompt
            }
        ]
        
        user_messages = [
            {
                "role": "user",
                "content": Prompts.ENTITY_FILTER_USER_PROMPT.format(**format_dict)
            }
        ]

        return system_messages + user_messages
    
    DUPLICATE_ENTITY_SYSTEM_PROMPT = """-Goal-
Consolidate Duplicate Entities in a List

-Steps-
1. You will be provided with a set of entities, each containing an entity name and an entity description.
2. Some entities in the list may refer to the same object or concept.
3. Combine the entities that describe the same object based on your common sense and understanding.
4. Return a list of entities, ensuring that there are no entities for the same object.
5. Additionally, provide the mapping relationship of each entity to the original entities [only provide the entity names].

-Note-
- Be sure to consider various ways entities might be referred to (e.g., abbreviations, different naming conventions).
- The output should be a cleaned list where each entity is distinct.
- The merging strategy should be as conservative as possible to avoid erroneous merges.

-Ouput Format-
{
    "entity A": ["original entity 1", "original entity 2", ...],
    xxx
}
"""

    @staticmethod
    def return_duplicate_entity_prompt(entity_list: List[str],
                                       system_prompt: str,
                                       need_few_shot: bool=True) -> str:
        
        entity_list_str = ", ".join([f'"{entity}"' for entity in entity_list])
        
        if not system_prompt:
            system_prompt = Prompts.DUPLICATE_ENTITY_SYSTEM_PROMPT
        
        system_messages = [
            {
                "role": "system",
                "content": system_prompt
            }
        ]
        
        user_messages = [
            {
                "role": "user",
                "content": f'Entity names: {entity_list_str}'
            }
        ]
        if not need_few_shot:
            return system_messages + user_messages
        
        few_shots = []

        return system_messages + few_shots + user_messages
    
    
    MERGE_ENTITY_SYSTEM_PROMPT = """- **Goal**:
  Given multiple dictionaries containing a `new_entity` with its name and description, along with a corresponding list of `candidate_entities` each consisting of entity name, a list of similar entity names, and each description, determine if the new entity matches any candidate entity based on either their name or the names in their similar entities list. 
  If a match is found, merge the `new_entity` into that candidate entity and specify the merged entity; otherwise, do not merge.

- **Steps**:
  1. **Input**:
      - `entities_data`: A list of dictionaries, where each dictionary contains:
        - `new_entity`: A dictionary with keys:
          - `name`: A string representing the new entity's name.
          - `description`: A string representing the new entity's description.
        - `candidate_entities`: A list of dictionaries, each containing:
          - `name`: A string representing the candidate entity's name.
          - `similar_names`: A list of strings representing similar names associated with the candidate entity.
          - `description`: A string representing the candidate entity's description.
  2. **Processing**:
      - For each dictionary in `entities_data`:
         - Check if the `new_entity`'s name matches the `name` or any name in the `similar_names` of each `candidate_entity`.
         - If a match is found:
             - Set `merged` to `true`.
             - Set `merge_target` to the matching entity name from `candidate_entities`.
             - Set `new_entity_name` to the original entity name.
         - If no match is found:
             - Set `merged` to `false`.
             - Set `merge_target` to `null`.
             - Set `new_entity_name` to the original entity name.
  3. **Output**:
      - Return a list of dictionaries, where each dictionary contains:
        - `merged`: A boolean indicating if a match was found (`true`) or not (`false`).
        - `merge_target`: The entity name of the matching candidate entity if `merged` is `True`, otherwise `null`.

- **Output Format**:
  ```json
  [
      {
          "merged": true/false,
          "merge_target": "candidate entity name" or null,
          "new_entity_name": "origin entity name"
      },
      ...
  ]
  ```
"""

    @staticmethod
    def return_merge_entity_prompt(merge_entity_json: str,
                                   system_prompt: str,
                                   need_few_shot: bool=True) -> str:
        
        if not system_prompt:
            system_prompt = Prompts.MERGE_ENTITY_SYSTEM_PROMPT
        
        system_messages = [
            {
                "role": "system",
                "content": system_prompt
            }
        ]
        
        user_messages = [
            {
                "role": "user",
                "content": merge_entity_json
            }
        ]
        if not need_few_shot:
            return system_messages + user_messages

        few_shots = [
            {
                "role": "system",
                "name": "example_user",
                "content": """entities_data = [
    {
        "new_entity": "alice.",
        "candidate_entities": ["Bob", "Alice", "Charlie"]
    },
    {
        "new_entity": "Daisy",
        "candidate_entities": ["Bob", "Alice", "Charlie"]
    }
]                
"""
            },
            {
                "role": "system",
                "name": "example_assistant",
                "content": """[
    {
        "merged": true,
        "merge_target": "Alice"
    },
    {
        "merged": false,
        "merge_target": null,
        "entities": ["Bob", "Alice", "Charlie"]
    }
]"""}
        ]
        return system_messages + few_shots + user_messages
    
    
    CONCEPT_EXTRACT_SYSTEM_PROMPT = """You are an intelligent and wise person, highly skilled in data analysis, capable of quickly extracting and uncovering the user's points of interest from their information. Additionally, you have strong interpersonal skills and can accurately, clearly, and efficiently describe your analysis results and convey your insights.**

Your task now is to extract some concepts from the user's private memories [Memory] that the user might be interested in or that appear frequently. These concepts should span multiple memories and reflect a common theme or phrase.**

Here are a few example concept names for reference:
- **Example 1**: Revolution of European Political Economy
- **Example 2**: Formula 1 racing car aerodynamics learning
- **Example 3**: Decoder-only transformers pretrain

The user will provide you with parts of their personal private memories [Memory], which may include:
- **Personal Creations**:
    These notes may record small episodes from the user's life, lyrical writings to express inner feelings, spontaneous essays inspired by thoughts, and even some meaningless content. 

- **Online Excerpts**:
    Information copied from the internet by the user that they may consider worth saving or may have saved on a whim.

Additionally, the user may provide you with a list of existing concepts they already have. 

1. If the user provides this list, your task is to first determine if there are any major concepts in the provided notes not already covered by the user's existing concepts. If so, you should extract these new concepts. Otherwise, do not deliberately extract similar concepts to avoid redundancy. Note that when there are many existing concepts, extracting new concepts should be done very cautiously.
   
2. If the user does not provide a list, you should carefully analyze the provided memories to extract potential concepts. These concepts should be as orthogonal as possible, with minimal overlap, and should represent different thematic information.

Your output structure should follow this format:
```json
[
    {
        "conceptName": "The name of the concept you extracted",
        "description": "Briefly explain why this concept is important, from which memories and data it was extracted, and describe the definition of this concept."
    },
    ...
]
```
"""
    
    CONCEPT_TIMELINE_IMPROVE_SYSTEM_PROMPT = """You are an intelligent and wise individual who excels at data analysis. 
You can quickly gain insights and extract valuable information from user-provided information. Additionally, you possess strong interpersonal skills, enabling you to clearly, efficiently, and accurately describe your analysis results and convey your insights.

I am assigning you a task where you will be given a concept [topic] that the user is interested in. Additionally, the user will provide you with a set of their personal memories [Memory]. You need to analyze and understand these memories.

For each memory, you need to determine whether there are any fragments related to the concept that interests the user. If such fragments exist, you need to extract a Concept Description from them. This Concept Description should include the following information:

1. Some specific attribute information of the concept in this memory.
2. A comprehensive description of the activities/roles/functions that the concept participates in during the memory.
3. The entities mentioned in the description should be described in the same language/spelling as used in the original memory.

The structure you need to return should be as follows:
```json
[
    {
        "refMemoryId": int (which memory this information came from),
        "description": str (Description of the concept),
        "createTime": str (Creation time of the memory, format: "YYYY-MM-DD HH:MM:SS")
    },
    ...
]
```
"""
    
    CONCEPT_DUPLICATE_SYSTEM_PROMPT = """- **Goal**:
Given multiple dictionaries containing a `New Concept` with its name and description, along with a corresponding list of `Candidate Concepts` each consisting of concept name, a list of similar concept names, and each description, determine if the new concept matches any candidate concept based on either their name or the names in their similar concepts list. 

If a match is found, merge the `New Concept` into that Candidate Concept and specify the merged concept; otherwise, do not merge.
- **Steps**:
  1. **Input**:
      - `Concept Group Data`: A list of dictionaries, where each dictionary contains:
          - `New Concept Name`: A string representing the new concept's name.
          - `New Concept Description`: A string representing the new concept's description.
        - `Candidate Concept List`: A list of dictionaries, each containing:
          - `Name`: A string representing the candidate concept's name.
          - `Similar Names`: A list of strings representing similar names associated with the candidate concept.
          - `Description`: A string representing the candidate concept's description.
  2. **Processing**:
      - For each dictionary in `concepts data`:
         - Check if the `New Concept`'s name matches the `name` or any name in the `Similar Names` of each `Candidate Concept`.
         - If a match is found:
             - Set `merged` to `true`.
             - Set `merge_target` to the matching concept name from `Candidate Concept List`.
             - Set `new_concept_name` to the original concept name.
         - If no match is found:
             - Set `merged` to `false`.
             - Set `merge_target` to `null`.
             - Set `new_concept_name` to the original concept name.
  3. **Output**:
      - Return a list of dictionaries, where each dictionary contains:
        - `merged`: A boolean indicating if a match was found (`true`) or not (`false`).
        - `merge_target`: The concept name of the matching candidate concept if `merged` is `True`, otherwise `null`.

- **Output Format**:
  ```json
  [
      {
          "merged": true/false,
          "merge_target": "candidate concept name" or null,
          "new_concept_name": "origin concept name"
      },
      ...
  ]
  ```
"""
    
    CONCEPT_CLUSTER_MERGE_SYSTEM_PROMPT = """You are an intelligent and wise person, highly skilled in data analysis, capable of swiftly extracting and identifying topics of interest from user information. Additionally, you possess excellent interpersonal skills, able to clearly, efficiently, and accurately describe your analysis results and convey your insights.

Now, I have a task for you. There are several clusters of memory that need to be merged. These clusters contain users' private memories, and you need to extract some concepts that are of interest to the users from this newly merged large cluster. These concepts often appear frequently and span multiple memories, effectively reflecting common themes or phrases.

Here are some example concepts you can use as reference:
Example 1: Revolution of European Political Economy
Example 2: Formula 1 racing car aerodynamics learning
Example 3: Pre-train for decoder-only transformers 

Additionally, the user will provide some previously extracted concepts from the original clusters, and you need to reference these to ensure the quality of the newly extracted concepts.

Your output structure should meet the following format:
[
    {
        "conceptName": "The name of the concept you extracted",
        "description": "A brief explanation of why this concept is important, from which memories and data it has been extracted, and a definition of the concept"
    },
    ...
]"""
    
    CONCEPT_EXTRACT_USER_PROMPT = """
### User's self-introduction ### [Can help you understand more about the user]
{user_self_intro}

### Currently extracted concepts ### [Do not extract the same/similar concepts again, to avoid repetition and redundancy.]
{extracted_concepts}

### User Memory List ### [Extract concepts from this part]
{memory_info_list}   
"""
    
    CONCEPT_TIMELIME_IMPROVE_USER_PROMPT = """
### Concept Information ###
{concept_info}

### Memory Information List ###
{memory_info_list}
"""
    
    CONCEPT_CLUSTER_MERGE_USER_PROMPT = """
### User's self-introduction ### [Can help you understand more about the user]
{user_self_intro}

### Previously Extracted Concepts ### [Reference these to ensure the quality of the newly extracted concepts]
{extracted_concepts}

### User Memory List ### [Extract concepts from this part]
{memory_info_list}
"""
    
    PERSONAL_WIKI_PERSON_SYSTEM_PROMPT = """You are an people analysis and document generation expert.

Below is a set of data that objectively explains the events that occurred at different times between this people and user. 
The content may include the definition of user and user's views on people {entity_name} or events that occurred. Now you need to focus on the description of the people {entity_name}. 


Generate a document that introduces the people {entity_name} to user from a second-person perspective. 
You should follow these steps to complete this task

1. ** Analysis and reasoning **
- Identify the primary and secondary of each content. The focus of this generation task is on the people {entity_name} and its relationship with user. 
- It is not necessary to generate the definition of user itself in the document. 
- Carefully analyze the content of the provided description list to form a thorough and multidimensional understanding of the mentioned people or people group.

2.** Review previous version document of {entity_name} **: 
- Based on the previous version document of {entity_name}, review the newly provided description information and think about the scope of the new description information. 
- Carefully adjust on the previous version of the document.

Preview Version of {entity_name}:
```
{preview_version_wiki}
```

3. ** Generate Documentation **
Based on the above steps, generate a document with detailed content, including details, and explain it to user from a second-person perspective.
The document should contain these parts:
1. The first part is about the result of how users view {entity_name} in their eyes. There is no need to objectively explain what kind of person {entity_name} is. The focus is on introducing how users view {entity_name}.
2. The second part is to infer the introduction of the target person through the provided events, and use specific events or nouns in the introduction to enhance persuasiveness. 
Do not have the same result as the first part, and improve the content quality of the document.

### Follow Instructions to Generate the Document
- If the document is about a person, keeping that person's name does not need to be translated into the corresponding language
- Only the report part is included in the generated content, without any additional analysis.
- Include the people names to provide full context.
- Written in {prefer_lang}.
- The name Felix and his desctiption is prohibited because you are writing in the second person 
### Output structure
## The relationshop with {entity_name}
xxx
##introduction
xx
"""
    PERSONAL_WIKI_PERSON_SYSTEM_PROMPT_gt_v0="""
You are an expert wiki writer specializing in capturing the nuances of a user's relationship with another individual. Your focus is on emotional and attitudinal dynamics, always maintaining a second - person perspective ("you/your") and adjusting depth based on the richness of the timeline data.

# Core Requirements:
- Strict second - person narrative (never use third person).
- Structured output with \n separators between sections.
- Concise analysis focused on essential relationship dynamics.
- Depth adjustment based on timeline data quantity.
- Minimal speculation, maximum factual synthesis.

# Input Processing:
- Evaluate data density: Rich details = deeper analysis | Sparse data = surface observations.
- Analyze {user_name}'s timeline entries about individual {entity_name}.
- Cross - reference with previous wiki (if provided): {preview_version_wiki}.
- Incorporate {self_intro} for tone/style guidance.

# Relationship Description:
- Generate a concise, engaging, and brief explanation of your relationship with the individual, -including its emotional and attitudinal aspects, all from a second - person perspective.
- Highlight your interactions with the individual, their significance, and the role they play in your context or experience.

# Style Enforcement:
- Use \n between sections, no markdown.
- Maximum 150 words total.
- Respond in {prefer_lang}.
- Individual's name inclusion in context.
- Speculation penalty: Only include inferences with 2+ timeline supports.
- Density adaptation: Expand/collapse sections based on data quantity.

# Prohibition List:
- Third - person references.
- Unsupported psychological assumptions.
- Timeline quantity mentions.
- Narrative flourishes without factual basis.

# Enhanced Expression Guidelines:
- Vary sentence structures to avoid repetitive beginnings.
- Use natural transitions between ideas.
- Incorporate subtle descriptive elements to enrich narrative flow.
- Maintain factual accuracy while enhancing readability.
- Employ diverse verbs and descriptive phrases to convey insights.
"""

    PERSONAL_WIKI_ENTITY_SYSTEM_PROMPT = """You are an entity analysis and document generation expert.

Below is a set of data that objectively explains the events that occurred at different times between this entity and user. 
The content may include the definition of user and user's views on entity {entity_name} or events that occurred. Now you need to focus on the description of the entity {entity_name}. 
Generate a document that introduces the entity {entity_name} to user from a second-person perspective. 

You should follow these steps to complete this task:

1. ** Analysis and reasoning **
Identify the primary and secondary of each content. 
The focus of this generation task is on the entity {entity_name} and its relationship with user.
It is not necessary to generate the definition of user itself in the document. 
Carefully analyze the content of the provided description list to form a thorough and multidimensional understanding of the mentioned entity or entity group.

2.** Review previous version document of {entity_name} **: 
Based on the previous version document of {entity_name}, review the newly provided description information and think about the scope of the new description information. 
Carefully adjust on the previous version of the document.

Preview Version Document of {entity_name}:
```
{preview_version_wiki}
```
3. ** Generate Documentation **
Based on the above steps, generate a document with detailed content, including details, and explain it to user from a second-person perspective.
The document should contain these parts. 
The first part is the relationship between the entity {entity_name} and user, It is not necessary to explain the definition of entities, but to focus on introducing their connections. 
The second part is a concept definition summarized through provided events in addition to the content summarized in the first part. 
The content of the second part should not overlap with the content of the first part.
It should be noted that when describing some positional entities, there is no need to generate any known introduction to the entity, but focus on the relationship with the entity and the user. 
Do not repeat the same results to improve the content quality of the document.

### Follow Instructions to Generate the Document
- If the entity is a person, keeping that person's name does not need to be translated into the corresponding language
- Only the report part is included in the generated content, without any additional analysis.
- Include the entity names to provide full context.
- Written in {prefer_lang}.
- The name Felix and user's role is prohibited because you are writing in the second person 
### Output structure
## The relationshop with me
xxx
## wikipedia
xx
"""
    PERSONAL_WIKI_ENTITY_SYSTEM_PROMPT_gt_v0 = """
You are an expert wiki writer specializing in crafting concise relationship analyses between users and entities. Always maintain a second-person perspective ("you/your") while dynamically adjusting depth based on timeline data richness.

**Core Requirements:**
1. Strict second-person narrative (never use third person)
2. Structured output with \n separators between sections
3. Concise analysis focused on essential relationship dynamics
4. Depth adjustment based on timeline data quantity
5. Minimal speculation, maximum factual synthesis

**Input Processing:**
- Analyze {user_name}'s timeline entries about {entity_name}
- Evaluate data density: Rich details = deeper analysis | Sparse data = surface observations
- Cross-reference with previous wiki (if provided): ```{preview_version_wiki}```
- Incorporate {self_intro} for tone/style guidance

**Relationship Description:**
- Generate a concise, engaging, and brief explanation of the entity, including its nature and its relation to the user, all from a second-person perspective.
- Highlight the user's interactions with the entity, its significance, and the role it plays in the user's context or experience from a second-person perspective.

**Style Enforcement:**
- Use \n between sections, no markdown
- Maximum 150 words total
- Respond in {prefer_lang}.
- Entity name inclusion in context
- Speculation penalty: Only include inferences with 2+ timeline supports
- Density adaptation: Expand/collapse sections based on data quantity

**Prohibition List:**
- Third-person references
- Unsupported psychological assumptions
- Timeline quantity mentions
- Narrative flourishes without factual basis

**Enhanced Expression Guidelines:**
- Vary sentence structures to avoid repetitive beginnings
- Use natural transitions between ideas
- Incorporate subtle descriptive elements to enrich narrative flow
- Maintain factual accuracy while enhancing readability
- Employ diverse verbs and descriptive phrases to convey insights
"""
     
    PERSONAL_WIKI_LOCATION_SYSTEM_PROMPT = """You are an entity analysis and document generation expert.

Below is a set of data that objectively explains the events that occurred at different times between this entity and user. 
The content may include the definition of user and user's views on entity {entity_name} or events that occurred. Now you need to focus on the description of the entity {entity_name}. 
Generate a document that introduces the entity {entity_name} to user from a second-person perspective. You should follow these steps to complete this task

1. ** Analysis and reasoning **
- Identify the primary and secondary of each content. 
- The focus of this generation task is on the entity {entity_name} and its relationship with user. 
- It is not necessary to generate the definition of user itself in the document. 
- Carefully analyze the content of the provided description list to form a thorough and multidimensional understanding of the mentioned entity or entity group.

2.** Review previous version document of {entity_name} **: 
Based on the previous version document of {entity_name}, review the newly provided description information and think about the scope of the new description information. Carefully adjust on the previous version of the document.

Preview Version Document of {entity_name}:
```
{preview_version_wiki}
```

3. ** Generate Documentation **
Based on the above steps, generate a document with detailed content, including details, and explain it to user from a second-person perspective.

The document only contain follow one part:
- This part is the relationship between the entity {entity_name} and user, It is not necessary to explain the definition of entities, but to focus on introducing their connections. 

### Follow Instructions to Generate the Document
- Only the report part is included in the generated content, without any additional analysis.
- Include the entity names to provide full context.
- Written in {prefer_lang}.
- The name {user_name} and his desctiption is prohibited because you are writing in the second person
 
### Output structure
##The relationshop with me
xxx
"""
    PERSONAL_WIKI_LOCATION_SYSTEM_PROMPT_gt_v0="""
    You are an expert wiki writer specializing in capturing the unique relationship between users and locations. Your focus is on emotional, experiential, and cultural dynamics, always maintaining a second - person perspective ("you/your") and adjusting depth based on the richness of the timeline data.

# Core Requirements:
- Strict second - person narrative (never use third person).
- Structured output with \n separators between sections.
- Concise analysis focused on essential relationship dynamics.
- Depth adjustment based on timeline data quantity.
- Minimal speculation, maximum factual synthesis.

# Input Processing:
- Analyze {user_name}'s timeline entries about location {entity_name}.
- Evaluate data density: Rich details = deeper analysis | Sparse data = surface observations.
- Cross - reference with previous wiki (if provided): {preview_version_wiki}.
- Incorporate {self_intro} for tone/style guidance.

# Relationship Description:
- Generate a concise, engaging, and brief explanation of the location, including its nature and its relation to you, all from a second - person perspective.
- Highlight your experiences with the location, its significance, and the role it plays in your context or life journey.

# Style Enforcement:
- Use \n between sections, no markdown.
- Maximum 150 words total.
- Respond in {prefer_lang}.
- Location name inclusion in context.
- Speculation penalty: Only include inferences with 2+ timeline supports.
- Density adaptation: Expand/collapse sections based on data quantity.

# Prohibition List:
- Third - person references.
- Unsupported psychological assumptions.
- Timeline quantity mentions.
- Narrative flourishes without factual basis.

# Enhanced Expression Guidelines:
- Vary sentence structures to avoid repetitive beginnings.
- Use natural transitions between ideas.
- Incorporate subtle descriptive elements to enrich narrative flow.
- Maintain factual accuracy while enhancing readability.
- Employ diverse verbs and descriptive phrases to convey insights. 
"""

    PERSONAL_WIKI_CONCEPT_SYSTEM_PROMPT = """You are an concept analysis and document generation expert.

Below is a set of data that objectively explains the events that occurred at different times between this concept and user. 
The content may include the definition of user and user's views on concept [{entity_name}] or events that occurred. Now you need to focus on the description of the concept [{entity_name}]. 
Generate a document that introduces the concept [{entity_name}] to user from a second-person perspective. You should follow these steps to complete this task

1. ** Analysis and reasoning **
- Identify the primary and secondary of each content. 
- The focus of this generation task is on the concept [{entity_name}] and its relationship with user. 
- It is not necessary to generate the definition of user itself in the document. 
- Carefully analyze the content of the provided description list to form a thorough and multidimensional understanding of the mentioned concept or concept group.

2.** Review previous version document of {entity_name} **: 
Based on the previous version document of {entity_name}, review the newly provided description information and think about the scope of the new description information. Carefully adjust on the previous version of the document.

Preview Version Document of {entity_name}:
```
{preview_version_wiki}
```

3. ** Generate Documentation **
Based on the above steps, generate a document with detailed content, including details, and explain it to user from a second-person perspective.

The document only contain follow one part:
- This part is the relationship between the concept [{entity_name}] and user, It is not necessary to explain the definition of concept, but to focus on introducing their connections. 

### Follow Instructions to Generate the Document
- Only the report part is included in the generated content, without any additional analysis.
- Include the concept names to provide full context.
- Written in {prefer_lang}.
- The name {user_name} and his desctiption is prohibited because you are writing in the second person
 
### Output structure
##The relationshop with me
xxx
"""

    TIMELINE_GENERATE_SYSTEM_PROMPT = """You are an insightful and intelligent assistant. Your current task is to organize a detailed and accurate evolution timeline for a specific entity. 
The user will provide information about the entity, including the entity's name and raw data extracted from various sources at different time points. 

Each piece of raw data is structured as follows:  ["CreateTime"]["SourceID"] "EntityDesc"
- "CreateTime" indicates the creation time of the raw data, reflecting the state or events experienced by the entity at that time.
- "SourceID" is a unique identifier for the source of the raw data
- "EntityDesc" describes specific matters related to the entity, all are extracted from the user's memory and may represent the user's views on the entity, as well as the user's interaction events with the entity.

Here are User Info:
{user_self_intro}

Your task is to complete the following steps:
1. First you to carefully read and deeply understand this information
2. Then try to explore the development history of the entity from the information of these entities
    - The development history of the relationship between the entity and user is recorded with each month as a single item, covering important events the entity participated in during that month, including origins, developments, turning points, major events, etc.
    - Each item should retain specific details as well as information about the time of occurrence.

3. Finally, organize it into a complete timeline
  - Organize the relationship between the entity and user's timeline based on the above development history, select requiring a clear development context, accurate event descriptions, specific details, arrange in reverse chronological order.
  - Remove the content in the Description that already exists in User Info to avoid information redundancy.
  - The description should not contain excessive user information; it should only describe the events related to the relationship between the user and the entity.
  - Do not use Entity as the subject; instead, adopt an objective perspective to introduce the events between the user and the Entity.
  Such as: xxxx, A xxx mention B xxx or xxx, A xxx with B xxx, A xxx said: B xxxx, and so on.
  - The description that defines an entity is the user's view or determination of that entity, maybe you can selectively add the relevant information, such as: A think that B is XXXXX.
  - Each description should be followed by the source information [source ID] that the description referenced.
  - Replace {user_name} in the description with "you" to reduce the sense of unfamiliarity with the user.

Output Format:

** YYYY-MM ** [Split by month]
- [YYYY-MM-DD] (Event Description, Attempt merge same day's events) [SourceID]
...

** YYYY-MM **
...

...


Do not use words like "possibly", "maybe" in the description, The information within the timeline should be certain and definite.
Please compile a complete and clear evolution timeline for the relationship between the entity and user.
write result in {prefer_lang}.
"""
    TIMELINE_GENERATE_SYSTEM_PROMPT_gt_v0="""
    You are a neutral and objective timeline compiler. Your role is to create a clear and concise timeline of the user({user_name})'s interactions with a specific entity {entity_name} based on their daily timelines. The focus should be on presenting the facts of the user's relationship with the entity in a straightforward manner, without adding unnecessary analysis or subjective interpretations.

** User Info:**
{user_self_intro}

** Your Task:**

1. ** Information Processing:**
   - Read the user's notes carefully and extract the key details about their interactions with the entity.
   - Do not make assumptions or draw conclusions beyond what is explicitly stated in the notes.

2. ** Timeline Creation:**
   - Organize the interactions by month, with each month as a separate section. 
   - For each day within a month, combine any similar or related interactions into a single entry. Only create separate entries for the same day if the content is significantly different.
   - You only need to extract information related to the entity {entity_name} and avoid other noise, and use the second person perspective to describe the user's actions and experiences with the entity. For example, "You used [entity] to...", "You encountered [entity] at...".
   - If you cannot extract entity-related information from the daily timeline, you can ignore the daily timeline instead of forcibly extracting it.
   - Avoid including any redundant information that is already present in the User Info section.
   - The timeline should be written in a simple and direct style, focusing on the user's objective experiences with the entity.

** Output Format:**
```
** YYYY-MM ** [Split by month]
- [YYYY-MM-DD] (Brief description of the interaction with the entity) [SourceID, number only]
...
** YYYY-MM **
...
```

** Guidelines:**

- Keep the descriptions short and to the point. Do not include any unnecessary details or elaborations.
- Use various expression techniques to make the output more natural and concise.
- Ensure that the timeline is easy to read and understand, with a clear chronological order.
- If there are multiple interactions on the same day, merge them into one entry unless they are about completely different aspects of the user's relationship with the entity.
- Write the timeline in {prefer_lang}.
"""

    MONTHLY_TIMELINE_TITLE_ENTITY_SYSTEM_PROMPT = """You are an observer of the evolution of concepts. A concept will develop over time, with many events happening every month. 
As these events go from nothing to something, the concept will go through different stages of development. Each stage of the event is based on new changes from the previous stage. 
Refer to the historical evolution process, infer the current stage of evolution, and generate an artistic title to describe this stage.

Output Format:
Date: [YYYY-MM]
Title:
{{Title}}

write result in {prefer_lang}.
Try to use the original name of the concept without translation.
"""

    MONTHLY_TIMELINE_TITLE_PERSON_SYSTEM_PROMPT = """You are an experienced private secretary. 
Users and people related to them will have some shared events, communication and conclusions, opinions on this person, and so on. 
This information can reflect the evolution of the relationship between the user and this person, as well as some help this person brings to the user. 
You need to refer to the historical evolution process and infer the current evolution stage based on the provided events. 
Write a title accourding to the story . 
This title reflects the impact and result it had on users during this period with detail below 10 words. 

Output Format:
Date: [YYYY-MM]
Title:
{{title}}

write title in {prefer_lang}.
Try to use the original name of the people without translation.
"""

    MONTHLY_TIMELINE_TITLE_LOCATION_SYSTEM_PROMPT = """You are an observer of the significance of location information to users. 
Users will experience events at a location that reflect the significance of the location to the user. As these events occur, the significance of the location to the user will go through different stages of development. 
Each stage is based on new changes from the previous stage. 
Based entirely on current events, referring to the historical evolution process, infer the current evolution stage and generate an art title to describe the changes in this stage before. 

Requirements for Title: 
1. Include some content to enhance credibility below 15 words.
2. Recent events should be summarized as higher-dimensional stages rather than enumerated.

Output Format:
Date: [YYYY-MM]
Title:
{{Title}}

write result in {prefer_lang}.
Try to use the original name of the concept without translation.
"""

    MONTHLY_TIMELINE_TITLE_CONCEPT_SYSTEM_PROMPT = """You are an observer of the evolution of concepts. A concept will develop over time, with many events happening every month. 
As these events go from nothing to something, the concept will go through different stages of development. Each stage of the event is based on new changes from the previous stage. 
Refer to the historical evolution process, infer the current stage of evolution, and generate an title to describe this stage.

Output Format:
Date: [YYYY-MM]
Title:
{{Title}}

write result in {prefer_lang}.
Try to use the original name of the concept without translation.
"""