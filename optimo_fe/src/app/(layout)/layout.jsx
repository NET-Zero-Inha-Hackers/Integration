import React from "react";
import Sidebar from "@/components/Sidebar";
import { ChatProvider } from "@/contexts/ChatContext";

export default function MainLayout({ children }) {
    return (
        <ChatProvider>
            <div className="flex flex-row">
                <Sidebar />

                {/* 메인 컨텐츠 */}
                <div className="flex-1 ml-20">
                    {children}
                </div>
            </div>
        </ChatProvider>
    );
}