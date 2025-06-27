import React from 'react';

export default function OverviewPage() {
    return (
        <div className="min-h-screen bg-[#23262b] text-white px-8 py-8">
            <div className="max-w-2xl mx-auto">
                <h2 className="text-base font-semibold mb-6">Service Overview</h2>
                {/* Optimo가 무엇인가요? */}
                <div className="mb-8">
                    <h3 className="text-2xl font-bold mb-2 flex items-center">
                        <span className="mr-2">🌍</span> Optimo가 무엇인가요?
                    </h3>
                    <p className="text-gray-200 text-base ml-7">
                        <span className="font-bold text-white">Optimo</span>는 프롬프트 난이도에 따라 가장 효율적인 AI 모델을 자동으로 선택하여,<br />
                        별도의 작업 없이 <span className="font-bold text-lime-400">AI 탄소 발자국</span>을 줄여주는 서비스입니다.
                    </p>
                </div>
                <hr className="border-gray-600 my-8" />
                {/* 핵심 기능 */}
                <div className="mb-8">
                    <h3 className="text-xl font-bold mb-4 flex items-center">
                        <span className="mr-2">🛠️</span> 핵심 기능
                    </h3>
                    <ul className="space-y-4 ml-7">
                        <li>
                            <span className="font-bold"><span className="mr-2">🧠</span>프롬프트 인텔리전스</span><br />
                            <span className="text-gray-300">프롬프트의 난이도를 분석해 최적의 AI 모델(<span className="font-bold text-white">Phi-3, GPT-4 Turbo</span> 등)로 자동 연결합니다.</span>
                        </li>
                        <li>
                            <span className="font-bold"><span className="mr-2">⚡</span>에너지 최적화</span><br />
                            <span className="text-gray-300">전력 소모를 최소화하고, 클라우드 에너지 비용을 최대 <span className="font-bold text-lime-400">70%</span>까지 절감할 수 있습니다.</span>
                        </li>
                        <li>
                            <span className="font-bold"><span className="mr-2">📊</span>임팩트 대시보드</span><br />
                            <span className="text-gray-300">AI 요청마다 절감된 전력량과 이산화탄소 감축량을 실시간으로 확인할 수 있습니다.</span>
                        </li>
                    </ul>
                </div>
                <hr className="border-gray-600 my-8" />
                {/* 왜 Optimo를 사용해야 하나요? */}
                <div className="mb-8">
                    <h3 className="text-xl font-bold mb-2 flex items-center">
                        <span className="mr-2">💡</span> 왜 Optimo를 사용해야 하나요?
                    </h3>
                    <p className="text-gray-200 text-base ml-7">
                        1인 개발자부터 대규모 AI 기업까지,<br />
                        <span className="font-bold text-white">Optimo</span>는 에너지를 절약하고 비용을 줄이며,<br />
                        환경을 생각하는 책임 있는 AI 활용을 가능하게 합니다.<br />
                        성능과 속도는 그대로 유지하면서 말이죠.
                    </p>
                </div>
            </div>
        </div>
    );
}
