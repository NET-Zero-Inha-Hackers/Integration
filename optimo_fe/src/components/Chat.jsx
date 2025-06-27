import React from 'react';
import Image from 'next/image';

export default function Chat({ title, description, onClick, isActive }) {
    return (
        <div
            className={
                "relative z-10 w-full px-4 pt-2 pb-4 rounded-lg cursor-pointer transform transition-transform duration-200 " +
                (isActive
                    ? "bg-gray-950 translate-x-1"
                    : "hover:bg-gray-950 hover:translate-x-1")
            }
            onClick={onClick}
        >
            <div className="flex items-center gap-2">
                <Image
                    src="/icon/optimo.png"
                    alt="Optimo Logo"
                    width={14}
                    height={14}
                />
                <p className="flex-1 text-sm font-bold truncate">{title}</p>
                {/* <Image
                    src="/icon/close.png"
                    alt="close icon"
                    width={16}
                    height={16}
                    onClick={() => { }}
                /> */}
            </div>
            <p className="text-xs text-gray-400 line-clamp-2 mt-1 ml-6">{description}</p>
        </div>
    );
}
