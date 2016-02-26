#pragma once
class BackgroundFeature
{
	public:
		virtual float getDistance(const BackgroundFeature &x) const = 0;
};