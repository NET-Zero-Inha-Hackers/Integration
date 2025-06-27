import Image from 'next/image';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import 'katex/dist/katex.min.css';
import { useAuth } from '@/contexts/AuthContext';

export default function ChatMessages({ chatList }) {
    const { user } = useAuth();
    const handleCopyText = (text) => {
        navigator.clipboard.writeText(text);
        alert("Copied to clipboard");
    };

    const markdownComponents = {
        table: ({node, ...props}) => (
            <table className="min-w-full border border-gray-500 border-collapse my-4" {...props} />
        ),
        th: ({node, ...props}) => (
            <th className="border border-gray-500 bg-[#23272f] px-3 py-2 font-bold text-center" {...props} />
        ),
        td: ({node, ...props}) => (
            <td className="border border-gray-500 px-3 py-2 text-center" {...props} />
        ),
        code({node, inline, className, children, ...props}) {
            if (inline) {
                return (
                    <code className="bg-[#23272f] rounded px-1.5 py-0.5 text-[#e0e0e0] text-xs" {...props}>
                        {children}
                    </code>
                );
            }
            return <code className="whitespace-pre-wrap" {...props}>{children}</code>;
        },
        pre: ({node, ...props}) => (
            <pre className="bg-[#181c23] rounded-lg p-2 my-2 overflow-hidden text-xs whitespace-pre-wrap" {...props} />
        ),
        h1: ({node, ...props}) => <h1 className="text-2xl font-bold mt-4 mb-2 text-[#e0e0e0]" {...props} />,
        h2: ({node, ...props}) => <h2 className="text-xl font-bold mt-3 mb-2 text-[#e0e0e0]" {...props} />,
        h3: ({node, ...props}) => <h3 className="text-lg font-semibold mt-2 mb-1 text-[#e0e0e0]" {...props} />,
        h4: ({node, ...props}) => <h4 className="text-base font-semibold mt-2 mb-1 text-[#e0e0e0]" {...props} />,
        h5: ({node, ...props}) => <h5 className="text-sm font-semibold mt-2 mb-1 text-[#e0e0e0]" {...props} />,
        h6: ({node, ...props}) => <h6 className="text-xs font-semibold mt-2 mb-1 text-[#e0e0e0]" {...props} />,
        blockquote: ({node, ...props}) => <blockquote className="border-l-4 border-[#6ee7b7] bg-[#23272f] pl-4 pr-2 py-2 my-2 text-[#b5e0d3] italic" {...props} />,
        hr: () => <hr className="my-4 border-t border-[#444]" />,
        strong: ({node, ...props}) => <strong className="font-bold text-[#fff]" {...props} />,
        em: ({node, ...props}) => <em className="italic text-[#b5b5b5]" {...props} />,
        del: ({node, ...props}) => <del className="line-through text-[#888]" {...props} />,
        div: ({node, ...props}) => <div {...props} />,
        p: ({node, ...props}) => <p {...props} />,
        li: ({node, ...props}) => <li {...props} />,
        a: ({node, ...props}) => <a className="text-blue-400 hover:text-blue-300" {...props} />,
    };

    if (!chatList || chatList.length === 0) {
        return (
            <div className="flex flex-col items-center justify-center h-full text-white">
                <div className="text-3xl font-bold mb-2">Optimo</div>
                <div className="text-base text-center">새로운 대화를 시작해볼까요?<br/>가장 효율적인 방식으로 응답해드립니다!</div>
            </div>
        );
    }
    return (
        <>
            {chatList.map((message, index) => (
                <div key={index} className={`flex mb-7 ${message.sender === 'USER' ? 'justify-end' : 'justify-start'}`}>
                    <div className="flex flex-row">
                        <Image src={message.sender === 'USER' ? '/userProfile/' + (parseInt(user.profileImage)+1) + '.png' : '/icon/ai.png'} alt="user" width={24} height={24} className="w-10 h-10 rounded-lg z-10" />
                        <div className="flex flex-col ml-2 max-w-full">
                            <div className="text-xs font-semibold">{message.sender}</div>
                            <div className={`flex flex-col text-xs rounded-2xl pr-4 pl-7 py-5 -ml-5 ${message.sender === 'USER' ? 'items-end bg-[#4b4f5b]' : 'items-start bg-[#28303F]'}`}>
                                {message.sender === 'AI' ? (
                                    <div className="prose prose-invert max-w-none text-xs w-full">
                                        <ReactMarkdown
                                            remarkPlugins={[remarkGfm, remarkMath]}
                                            rehypePlugins={[rehypeKatex]}
                                            components={markdownComponents}
                                        >
                                            {message.text}
                                        </ReactMarkdown>
                                    </div>
                                ) : (
                                    <div>{message.text}</div>
                                )}
                                <div className={`flex flex-row w-full items-center justify-between mt-8 ${message.sender === 'AI' ? '' : 'hidden'}`}>
                                    <div className="text-xs text-[#888888]">Model: {message.model}</div>
                                    <button 
                                        onClick={() => handleCopyText(message.text)}
                                        className="text-xs text-[#888888] hover:text-[#aaaaaa] transition-colors flex flex-row items-center bg-[#202633] rounded-lg min-h-7 px-4 gap-4"
                                    >
                                        <Image src="/icon/copy.svg" alt="copy" width={10} height={10} />
                                        Copy
                                    </button>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            ))}
        </>
    );
} 