'use client';
import React, { useEffect, useState } from 'react';
import Chat from '@/components/Chat';
import Image from 'next/image';
import { useRouter } from 'next/navigation';
import { useChat } from '@/contexts/ChatContext';

export default function Chatlist() {
    const router = useRouter();
    const { allChats, currentChatId, setCurrentChatId, loading } = useChat();
    const [lastViewedChatId, setLastViewedChatId] = useState(null);

    useEffect(() => {
        const savedId = localStorage.getItem("lastChattingId");
        if (savedId) {
            setLastViewedChatId(savedId);
            setCurrentChatId(savedId);
        }
    }, [setCurrentChatId]);

    const handleChatClick = (chatId) => {
        localStorage.setItem("lastChattingId", chatId);
        setLastViewedChatId(chatId);
        setCurrentChatId(chatId);
        router.push('/chat/' + chatId);
    };

    const handleNewChat = () => {
        localStorage.setItem("lastChattingId", "");
        setLastViewedChatId("");
        setCurrentChatId(null);
        if (window.location.pathname === '/chat') {
            window.location.reload();
        } else {
            router.push('/chat');
        }
    };

    if (loading) {
        return (
            <div className="flex flex-col w-80 h-screen">
                <div className="flex items-center justify-between p-4 mb-2 w-full">
                    <h2 className="text-xl font-bold text-left">My Chats</h2>
                </div>
                <div className="flex items-center justify-center h-full">
                    <div className="text-white">로딩 중...</div>
                </div>
            </div>
        );
    }

    return (
        <div className="flex flex-col w-80 h-screen overflow-x-visible">
            <div className="flex items-center justify-between p-4 mb-2 w-full">
                <h2 className="text-xl font-bold text-left">My Chats</h2>
                <Image
                    src="/icon/newChat.png"
                    alt="newChat Icon"
                    width={32}
                    height={32}
                    className="cursor-pointer"
                    onClick={handleNewChat}
                />
            </div>

            <div className="relative h-[85vh] overflow-y-scroll overflow-x-visible px-1 pt-1">
                <div className="relative z-0">
                    {Array.isArray(allChats) && [...allChats].reverse().map((chat) => (
                        <Chat
                            key={chat._id}
                            title={chat.title}
                            description={chat.description}
                            onClick={() => handleChatClick(chat._id)}
                            isActive={chat._id === lastViewedChatId}
                        />
                    ))}
                </div>
            </div>
        </div>
    );
}