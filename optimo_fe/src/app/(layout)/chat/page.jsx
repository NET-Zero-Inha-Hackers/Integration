"use client";
import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import Chatlist from "@/components/Chatlist";
import Image from "next/image";
import ChatMessages from "@/components/ChatMessages";
import { newChatWebSocket, continueChatWebSocket } from "@/lib/websocket";
import { useAuth } from "@/contexts/AuthContext";
import { useChat } from "@/contexts/ChatContext";

export default function MainPage() {
  const router = useRouter();
  const { user, isLoading: authLoading } = useAuth();
  const { addChat, setCurrentChatId, updateChat } = useChat();
  const [isRedirecting, setIsRedirecting] = useState(true);
  const [chat, setChat] = useState({
    chattingId: "",
    ownerId: "",
    title: "",
    description: "",
    chatList: [],
    createdAt: 0,
    modifiedAt: 0,
  });

  useEffect(() => {
    if (!authLoading && !user) {
      router.push('/');
      return;
    }

    const lastChattingId = localStorage.getItem("lastChattingId");
    console.log("lastChattingId",lastChattingId);

    if (lastChattingId) {
      router.push(`/chat/${lastChattingId}`);
    } else {
      setIsRedirecting(false);
    }
  }, [router, user, authLoading]);

  // 채팅 목록이 변경될 때마다 자동으로 스크롤
  useEffect(() => {
    const chatBox = document.getElementById('chat-box');
    if (chatBox) {
      setTimeout(() => {
        chatBox.scrollTo({
          top: chatBox.scrollHeight,
          behavior: 'smooth'
        });
      }, 100);
    }
  }, [chat.chatList]);

  const [formData, setFormData] = useState({
    content: "",
  });

  const handleInputChange = (e) => {
    const { name, value } = e.target;

    e.target.style.height = "auto";
    const newHeight = Math.min(200, e.target.scrollHeight);
    e.target.style.height = newHeight + 5 + "px";

    setFormData((prev) => ({
      ...prev,
      [name]: value,
    }));
  };

  const handleSendMessage = () => {
    if (formData.content.trim() === "") return;

    const userChatElement = {
      sender: "USER",
      text: formData.content,
      timestamp: Math.floor(Date.now() / 1000),
      model: "",
      use_estimate: 0,
      llm_estimate: 0,
    };

    setChat((prev) => ({
      ...prev,
      chatList: [...prev.chatList, userChatElement],
    }));

    function modelNameHandler(modelName) {
      const aiChatElement = {
        sender: "AI",
        text: "",
        timestamp: Math.floor(Date.now() / 1000),
        model: modelName,
        use_estimate: 0,
        llm_estimate: 0,
      };
      setChat((prev) => ({
        ...prev,
        chatList: [...prev.chatList, aiChatElement],
      }));
      console.log("모델 이름:", modelName);
    }

    function modelResponseHandler(id, text) {
      setChat((prev) => {
        const updatedChatList = [...prev.chatList];
        const lastIndex = updatedChatList.length - 1;
        if (updatedChatList[lastIndex] && updatedChatList[lastIndex].sender === "AI") {
          updatedChatList[lastIndex].text += text;
        }
        return {
          ...prev,
          chatList: updatedChatList,
        };
      });

      console.log("모델 응답:", id, text);
      const chatBox = document.getElementById('chat-box');
      if (chatBox) {
        setTimeout(() => {
          chatBox.scrollTo({
            top: chatBox.scrollHeight,
            behavior: 'smooth'
          });
        }, 100);
      }
    }

    async function metadataHandler(id, title, description) {
      const newChatData = {
        _id: id,
        chattingId: id,
        ownerId: user?.id || "",
        title: title,
        description: description,
        chatList: chat.chatList,
        createdAt: Date.now(),
        modifiedAt: Date.now(),
      };

      // Context에 새 채팅 추가
      addChat(newChatData);
      
      // 현재 채팅 설정
      setCurrentChatId(id);
      
      setChat((prev) => ({
        ...prev,
        chattingId: id,
        title: title,
        description: description,
      }));
      
      localStorage.setItem("lastChattingId", id);
      console.log("메타데이터:", id, title, description);
    }

    const token = document.cookie.split('; ').find(row => row.startsWith('jwtToken='))?.split('=')[1];

    if (chat.chattingId && chat.chatList.length > 0) {
      // 기존 채팅에 메시지 추가
      const socket = continueChatWebSocket(
        chat.chattingId,
        formData.content,
        token,
        modelNameHandler,
        modelResponseHandler
      );
    } else {
      // 새 채팅 생성
      const socket = newChatWebSocket(
        formData.content,
        token,
        modelNameHandler,
        modelResponseHandler,
        metadataHandler
      );
    }

    setFormData({ content: "" });
    const textarea = document.querySelector('textarea[name="content"]');
    if (textarea) textarea.style.height = "auto";
  };

  const handleInputKeyDown = (e) => {
    if (e.isComposing || (e.nativeEvent && e.nativeEvent.isComposing)) return;
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  if (isRedirecting) {
    return (
      <div className="flex flex-row h-screen">
        <Chatlist />
        <div className="flex-1 flex items-center justify-center">
          <div className="text-white">로딩 중...</div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-row h-screen">
      <Chatlist />
      <div className="flex-1 flex flex-col items-start justify-start p-4">
        <div className="text-2xl font-bold pl-6 mb-2">
          {chat.title == "" ? "New Chat" : chat.title}
        </div>
        <div className="flex flex-col w-full flex-1 bg-[#3F424A] rounded-lg p-8 min-h-0">
          <div
            id="chat-box"
            className="flex flex-col flex-1 overflow-y-auto min-h-0"
          >
            <ChatMessages chatList={chat.chatList} />
          </div>
          <div
            id="input-box"
            className="flex flex-row items-center justify-between bg-[#4b4f5b] rounded-lg text-[#eeeeee] min-h-14 shadow-lg"
          >
            <textarea
              name="content"
              value={formData.content}
              onChange={handleInputChange}
              onKeyDown={handleInputKeyDown}
              className="w-full px-4 py-2 rounded-md resize-none overflow-y-auto bg-transparent outline-none max-h-[150px]"
              style={{ maxHeight: "150px" }}
              placeholder="내용을 입력하세요..."
              rows={1}
            />
            <button
              className="p-2 hover:bg-[#5b5f6b] rounded-lg transition-colors"
              onClick={handleSendMessage}
            >
              <Image src="/icon/arrow.svg" alt="전송" width={28} height={28} />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
