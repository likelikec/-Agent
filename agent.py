import json
from dataclasses import dataclass
from typing import List, Dict, Set, Optional
from enum import Enum
from contextlib import contextmanager
from openai import OpenAI



# 自定义异常
class CourseError(Exception):
    pass


class APIError(Exception):
    pass


class ValidationError(Exception):
    pass


# 枚举操作类型
class Action(Enum):
    QUERY = "query"
    SELECT = "select"
    DELETE = "delete"
    SHOW = "show_selected"
    ERROR = "error"


class ConversationManager:
    def __init__(self, max_history: int = 3):
        self.history = []
        self.max_history = max_history

    def add_message(self, role: str, content: str):
        self.history.append({"role": role, "content": content})
        if len(self.history) > self.max_history:
            self.history.pop(0)

    def get_recent_history(self) -> str:
        return "\n".join(
            f"{'学生' if msg['role'] == 'user' else '助手'}: {msg['content']}"
            for msg in self.history
        )


class OpenAIClient:
    def __init__(self, api_key: str, base_url: str):
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def get_completion(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        except Exception as e:
            raise APIError(f"API调用失败: {str(e)}")


class CourseSystem:
    def __init__(self):
        self.courses = {
            "高等数学": {"type": "必修", "tags": ["理科", "基础课"]},
            "线性代数": {"type": "必修", "tags": ["理科", "基础课"]},
            "数据结构与算法": {"type": "必修", "tags": ["编程", "专业核心"]},
            "计算机网络": {"type": "必修", "tags": ["专业核心", "网络"]},
            "操作系统": {"type": "必修", "tags": ["专业核心", "系统"]},
            "概率论与数理统计": {"type": "必修", "tags": ["理科", "基础课"]},
            "信号与系统": {"type": "必修", "tags": ["专业基础", "通信"]},
            "数字信号处理": {"type": "必修", "tags": ["专业核心", "通信"]},
            "通信原理": {"type": "必修", "tags": ["专业核心", "通信"]},
            "电磁场与电磁波": {"type": "必修", "tags": ["专业基础", "通信"]},
            "微波技术与天线": {"type": "必修", "tags": ["专业核心", "通信"]},
            "人工智能导论": {"type": "选修", "tags": ["专业方向", "AI"]},
            "云计算技术": {"type": "选修", "tags": ["专业方向", "云计算"]},
            "大数据处理": {"type": "选修", "tags": ["专业方向", "数据"]},
            "现代交换技术": {"type": "必修", "tags": ["专业核心", "通信"]},
            "光纤通信": {"type": "必修", "tags": ["专业核心", "通信"]},
            "移动通信": {"type": "必修", "tags": ["专业核心", "通信"]},
            "卫星通信": {"type": "选修", "tags": ["专业方向", "通信"]},
            "无线网络": {"type": "选修", "tags": ["专业方向", "通信"]},
            "通信网安全": {"type": "选修", "tags": ["专业方向", "安全"]},
            "移动应用开发": {"type": "选修", "tags": ["编程", "移动开发"]},
            "信息安全": {"type": "选修", "tags": ["专业方向", "安全"]},
            "信息安全基础": {"type": "必修", "tags": ["专业核心", "安全"]},
            "密码学": {"type": "必修", "tags": ["专业核心", "安全"]},
            "网络安全技术": {"type": "必修", "tags": ["专业核心", "安全"]},
            "系统安全": {"type": "必修", "tags": ["专业核心", "安全"]},
            "应用安全": {"type": "必修", "tags": ["专业核心", "安全"]},
            "网络攻防技术": {"type": "必修", "tags": ["专业核心", "安全"]},
            "网络安全法律法规": {"type": "选修", "tags": ["专业方向", "法律"]},
            "云计算安全": {"type": "选修", "tags": ["专业方向", "云计算", "安全"]},
            "体育": {"type": "选修", "tags": ["运动"]}
        }
        self.selected_courses = set()
        self.conversation_manager = ConversationManager()
        self.last_mentioned_course = None
        self.ai_client = OpenAIClient(
            api_key="your key",
            base_url="https://www.aigptx.top/v1"
        )

    @contextmanager
    def session_context(self):
        try:
            yield
        except CourseError as e:
            print(f"课程操作错误: {str(e)}")
        except APIError as e:
            print(f"API错误: {str(e)}")
        except Exception as e:
            print(f"系统错误: {str(e)}")

    def find_similar_courses(self, course_name: str, selected: bool = False) -> List[str]:
        """查找相似的课程名称"""
        if selected:
            courses = self.selected_courses
        else:
            courses = self.courses.keys()

        # 完全匹配
        if course_name in courses:
            return [course_name]

        similar_courses = []
        search_term = course_name.lower()

        # 模糊匹配
        for course in courses:
            course_lower = course.lower()
            # 包含关系
            if search_term in course_lower or course_lower in search_term:
                similar_courses.append(course)
            # 关键词匹配
            elif any(keyword in course_lower for keyword in search_term.split()):
                similar_courses.append(course)

        return similar_courses

    def select_course(self, course_name: str, force: bool = False) -> dict:
        """选择课程，返回结果字典"""
        result = {
            "success": False,
            "message": "",
            "similar_courses": [],
            "needs_confirmation": False,
            "course_to_confirm": None
        }

        # 如果是强制选课模式，直接尝试选择
        if force and course_name in self.courses:
            if course_name in self.selected_courses:
                result["message"] = f"您已经选择了 {course_name}"
            else:
                self.selected_courses.add(course_name)
                result["success"] = True
                result["message"] = f"成功选择 {course_name}"
            return result

        # 查找相似课程
        similar_courses = self.find_similar_courses(course_name)

        if not similar_courses:
            result["message"] = f"未找到与 '{course_name}' 相关的课程"
            return result

        if len(similar_courses) == 1:
            found_course = similar_courses[0]
            result["needs_confirmation"] = True
            result["course_to_confirm"] = found_course
            result["message"] = f"您是否想选择 '{found_course}' ？"
            return result

        result["similar_courses"] = similar_courses
        result["message"] = f"找到多个相关课程：\n" + \
                            "\n".join(f"{i + 1}. {c}" for i, c in enumerate(similar_courses))
        return result

    def delete_course(self, course_name: str, force: bool = False) -> dict:
        """退选课程，返回结果字典"""
        result = {
            "success": False,
            "message": "",
            "similar_courses": [],
            "needs_confirmation": False,
            "course_to_confirm": None
        }

        # 如果是强制退选模式，直接尝试退选
        if force and course_name in self.selected_courses:
            self.selected_courses.remove(course_name)
            result["success"] = True
            result["message"] = f"成功退选 {course_name}"
            return result

        # 首先检查是否在已选课程中
        if course_name in self.selected_courses:
            result["needs_confirmation"] = True
            result["course_to_confirm"] = course_name
            result["message"] = f"确认要退选 '{course_name}' 吗？"
            return result

        # 查找相似的已选课程
        similar_courses = self.find_similar_courses(course_name, selected=True)

        if not similar_courses:
            result["message"] = f"未找到与 '{course_name}' 相关的已选课程"
            return result

        if len(similar_courses) == 1:
            found_course = similar_courses[0]
            result["needs_confirmation"] = True
            result["course_to_confirm"] = found_course
            result["message"] = f"您是否想退选 '{found_course}' ？"
            return result

        result["similar_courses"] = similar_courses
        result["message"] = f"找到多个相关的已选课程：\n" + \
                            "\n".join(f"{i + 1}. {c}" for i, c in enumerate(similar_courses))
        return result

    def validate_course_name(self, course_name: str) -> bool:
        if not course_name or not isinstance(course_name, str):
            raise ValidationError("课程名称无效")
        return course_name in self.courses

    def query_courses(self, filters: List[str] = None, interests: List[str] = None) -> List[dict]:
        results = []
        for course_name, info in self.courses.items():
            if not filters or info["type"] in filters:
                results.append({
                    "name": course_name,
                    "type": info["type"],
                    "tags": info["tags"]
                })

        if interests:
            results.sort(
                key=lambda x: len(set(self.courses[x["name"]]["tags"]) & set(interests)),
                reverse=True
            )

        return results

    def show_selected_courses(self) -> List[Dict]:
        if not self.selected_courses:
            return "您还没有选择任何课程"

        return [
            {
                "name": course,
                "type": self.courses[course]["type"],
                "tags": self.courses[course]["tags"]
            }
            for course in self.selected_courses
        ]



    def parse_user_input(self, user_input: str) -> dict:
        self.conversation_manager.add_message("user", user_input)

        prompt = self._build_prompt(user_input)

        try:
            response = self.ai_client.get_completion(prompt)
            parsed_response = json.loads(response)

            self.conversation_manager.add_message(
                "assistant",
                parsed_response["context"]["message"]
            )

            if parsed_response.get("course_name"):
                self.last_mentioned_course = parsed_response["course_name"]

            return parsed_response

        except json.JSONDecodeError:
            raise APIError("无法解析API响应")

    def _build_prompt(self, user_input: str) -> str:
        conversation = self.conversation_manager.get_recent_history()

        return f"""你是一个智能课程助手，帮助学生选择或管理他们的课程。请理解学生的自然语言输入并给出合适的响应。

当前可用的课程信息：
{json.dumps(self.courses, ensure_ascii=False, indent=2)}

学生已选课程：{list(self.selected_courses)}
上次提到的课程：{self.last_mentioned_course}

{conversation}

请以对话的方式理解学生的意图，注意以下几点：

1.学生可能会用非正式的表达方式，例如"我不想上课"或"我想去运动"等。
2.根据学生的表达，推测他们可能感兴趣的课程。
3.如果学生提到某个领域（如运动），请推荐相关的课程。
4.处理指代词（如"它"或"这个"）时，应参考上下文中最后提到的课程。
5.如果学生没有明确的意图，请推荐最适合的课程。
6.在选课或退课时，根据上下文判断是否需要确认，并设置needs_confirmation。
7.如果学生想查看已选课程，请返回show_selected动作。

请返回以下JSON格式的响应，注意只返回JSON代码，不要任何额外内容，不要用代码块包裹：
{{
    "action": "query/select/delete/show_selected",  # 查询、选课、退课或查看已选
    "filters": ["必修"/"选修"],      # 可选，课程类型过滤
    "course_name": "具体课程名称",    # 从可用课程中选择，查询时可为null
    "interests": ["用户兴趣标签"],    # 用户表达的兴趣方向
    "context": {{
        "is_certain": true/false,     # 是否确定用户意图
        "needs_confirmation": false,  # 是否需要确认
        "suggestions": [],           # 建议的操作
        "message": ""               # 返回给学生的信息
    }}
}}

只返回JSON代码！不要用代码块包裹！"""


def main():
    system = CourseSystem()

    while True:
        user_input = input('\n请输入您的需求(输入"退出"结束):')
        if user_input == "退出":
            break

        with system.session_context():
            parsed = system.parse_user_input(user_input)

            if parsed["action"] == Action.ERROR.value:
                continue

            if parsed.get("context", {}).get("message"):
                print(parsed["context"]["message"])

            if parsed["action"] == Action.QUERY.value:
                results = system.query_courses(parsed.get("filters"), parsed.get("interests"))
                if results:
                    print("\n根据您的兴趣，为您推荐以下课程:")
                    for course in results:
                        status = "已选" if course["name"] in system.selected_courses else "未选"
                        print(f"  - {course['name']} [{course['type']}] {status}")
                        print(f"    标签: {', '.join(course['tags'])}")

                    # 提示用户是否要选择课程
                    print("\n是否想选择以上推荐的课程？请输入课程编号进行选择，或输入 n 取消:")
                    choice = input("请选择(1-{0}): ".format(len(results)))
                    if choice.isdigit() and 1 <= int(choice) <= len(results):
                        selected_course = results[int(choice) - 1]["name"]
                        result = system.select_course(selected_course, force=True)
                        print(result["message"])
                else:
                    print("抱歉，没有找到符合您兴趣的课程")

            elif parsed["action"] == Action.SELECT.value:
                if not parsed.get("course_name"):
                    print("请指定要选择的具体课程")
                    continue

                result = system.select_course(parsed["course_name"])
                print(result["message"])

                if result["needs_confirmation"]:
                    confirm = input("请确认是否选择该课程 (y/n): ")
                    if confirm.lower() == 'y':
                        final_result = system.select_course(result["course_to_confirm"], force=True)
                        print(final_result["message"])

                elif result["similar_courses"]:
                    choice = input("请输入课程编号选择课程，或输入 n 取消: ")
                    if choice.isdigit() and 1 <= int(choice) <= len(result["similar_courses"]):
                        selected_course = result["similar_courses"][int(choice) - 1]
                        final_result = system.select_course(selected_course, force=True)
                        print(final_result["message"])

            elif parsed["action"] == Action.DELETE.value:
                if not parsed.get("course_name"):
                    print("请指定要退选的具体课程")
                    continue

                result = system.delete_course(parsed["course_name"])
                print(result["message"])

                if result["needs_confirmation"]:
                    confirm = input("请确认是否退选该课程 (y/n): ")
                    if confirm.lower() == 'y':
                        final_result = system.delete_course(result["course_to_confirm"], force=True)
                        print(final_result["message"])

                elif result["similar_courses"]:
                    choice = input("请输入课程编号选择要退选的课程，或输入 n 取消: ")
                    if choice.isdigit() and 1 <= int(choice) <= len(result["similar_courses"]):
                        selected_course = result["similar_courses"][int(choice) - 1]
                        final_result = system.delete_course(selected_course, force=True)
                        print(final_result["message"])

            elif parsed["action"] == Action.SHOW.value:
                results = system.show_selected_courses()
                if isinstance(results, str):
                    print(results)
                else:
                    print("\n您已选择的课程:")
                    for course in results:
                        print(f"  - {course['name']} [{course['type']}]")
                        print(f"    标签: {', '.join(course['tags'])}")


if __name__ == "__main__":
    main()
