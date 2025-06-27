"use client";
export const runtime = "edge";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import Chatlist from "@/components/Chatlist";
import ChatMessages from "@/components/ChatMessages";
import Image from "next/image";
import { useChat } from "@/contexts/ChatContext";
import { continueChatWebSocket } from "@/lib/websocket";

export default function ChatPage() {
  const params = useParams();
  const chattingId = params.chattingId;
  const { getCurrentChat, updateChat, setCurrentChatId } = useChat();
  
  const [formData, setFormData] = useState({
    content: "",
  });

  // 실시간 스트리밍을 위한 로컬 상태
  const [localChatList, setLocalChatList] = useState([]);

  // chattingId가 바뀔 때마다 현재 채팅 설정
  useEffect(() => {
    setCurrentChatId(chattingId);
  }, [chattingId, setCurrentChatId]);

  // 현재 채팅 데이터 가져오기
  const chatData = getCurrentChat();

  // Context 데이터가 변경되면 로컬 상태 동기화
  useEffect(() => {
    if (chatData?.chatList) {
      setLocalChatList([...chatData.chatList]);
    }
  }, [chatData?.chatList]);

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
  }, [localChatList]);

  const handleInputChange = (e) => {
    const { name, value } = e.target;

    // 자동 높이 조절
    e.target.style.height = "auto";
    const newHeight = Math.min(200, e.target.scrollHeight);
    e.target.style.height = newHeight + 5 + "px";

    setFormData((prev) => ({
      ...prev,
      [name]: value,
    }));
  };

  const handleSendMessage = () => {
    if (formData.content.trim() === "" || !chatData) return;

    console.log("=== 메시지 전송 시작 ===");

    const userChatElement = {
      sender: "USER",
      text: formData.content,
      timestamp: Math.floor(Date.now() / 1000),
      model: "",
      use_estimate: 0,
      llm_estimate: 0,
    };

    // 1. 유저 메시지를 Context에 즉시 저장
    const updatedChatListWithUser = [...(chatData.chatList || []), userChatElement];
    updateChat(chattingId, {
      chatList: updatedChatListWithUser
    });

    let currentAIMessage = null; // 현재 AI 메시지 추적

    function modelNameHandler(modelName) {
      console.log("AI 모델 시작:", modelName);
      
      // AI 메시지 박스 생성 (로컬 상태에만)
      currentAIMessage = {
        sender: "AI",
        text: "",
        timestamp: Math.floor(Date.now() / 1000),
        model: modelName,
        use_estimate: 0,
        llm_estimate: 0,
      };
      
      setLocalChatList(prev => [...prev, currentAIMessage]);
    }

    function modelResponseHandler(id, text) {
      // 실시간 스트리밍 (로컬 상태만 업데이트)
      if (currentAIMessage) {
        currentAIMessage.text += text;
        setLocalChatList(prev => {
          const newList = [...prev];
          const lastIndex = newList.length - 1;
          if (newList[lastIndex] && newList[lastIndex].sender === "AI") {
            newList[lastIndex] = { ...currentAIMessage };
          }
          return newList;
        });
      }
    }

    // WebSocket 연결 설정 (응답 완료 처리 추가)
    const socket = continueChatWebSocket(
      chattingId,
      formData.content,
      document.cookie.split('; ').find(row => row.startsWith('jwtToken='))?.split('=')[1],
      modelNameHandler,
      modelResponseHandler
    );

    // WebSocket 메시지 이벤트 추가 처리
    const originalOnMessage = socket.onmessage;
    socket.onmessage = (event) => {
      const data = JSON.parse(event.data);
      
      // 기존 처리
      if (originalOnMessage) {
        originalOnMessage(event);
      }
      
      // 응답 완료 시 Context에 저장
      if (data.end === "end" && currentAIMessage) {
        console.log("AI 응답 완료 - Context에 저장");
        const finalChatList = [...updatedChatListWithUser, currentAIMessage];
        updateChat(chattingId, {
          chatList: finalChatList,
          modifiedAt: Date.now()
        });
      }
    };

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

  if (!chatData) {
    return (
      <div className="flex flex-row h-screen">
        <Chatlist />
        <div className="flex-1 flex items-center justify-center">
          <div className="text-white">채팅을 찾을 수 없습니다</div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-row h-screen">
      <Chatlist />
      <div className="flex-1 flex flex-col items-start justify-start p-4">
        <div className="text-2xl font-bold pl-6 mb-2">{chatData.title}</div>
        <div className="flex flex-col w-full flex-1 bg-[#3F424A] rounded-lg p-8 min-h-0">
          <div
            id="chat-box"
            className="flex flex-col flex-1 overflow-y-auto min-h-0"
          >
            <ChatMessages chatList={localChatList} />
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
