import scrapy


class TripAdvisorSpider(scrapy.Spider):
    name = "tripAdvisor"
    allowed_domains = ['www.tripadvisor.com']
    #start_urls = [
    #    "http://www.tripadvisor.com/Attractions-g60763-Activities-a_allAttractions.true-New_York_City_New_York.html"
    #]
    #start_urls = [
    #    "https://www.tripadvisor.com/Attractions-g55197-Activities-a_allAttractions.true-Memphis_Tennessee.html"
    #]
    #start_urls = [
    #    "https://www.tripadvisor.com/Attractions-g35805-Activities-a_allAttractions.true-Chicago_Illinois.html"
    #]
    #start_urls = [
    #    "https://www.tripadvisor.com/Attractions-g34438-Activities-a_allAttractions.true-Miami_Florida.html"
    #]
    start_urls = [
        "https://www.tripadvisor.com/Attractions-g60750-Activities-a_allAttractions.true-San_Diego_California.html"
    ]
    #start_urls = [
    #    "https://www.tripadvisor.com/Attractions-g28970-Activities-a_allAttractions.true-Washington_DC_District_"
    #    "of_Columbia.html"
    #]

    # Record PoIs reviews page already extracted
    poiUrl = set()
    # Record reviews already extracted
    reviewUrl = set()
    # Record PoIs already extracted
    poiSet = set()

    def parse(self, response):
        # Redirect to Reviews page
        poiLinks = response.xpath('//a[@class="_2OD1jgdZ"]/@href').extract()

        for link in poiLinks:
            # Generate absolute url link for the review of each PoI
            poiAbsPath = response.urljoin(link + "#REVIEWS")

            if poiAbsPath not in self.poiUrl:
                self.poiSet.add(link)

                self.poiUrl.add(poiAbsPath)

                yield scrapy.Request(poiAbsPath, callback=self.parseReviewsSum)

        # a - 1 ; a30 - 2; a60 - 3
        #curPage = int(response.xpath('//div[@class="pageNumbers"]/span[contains(@class, "pageNum current")]/text()').
        #              extract_first())

        curPage = response.xpath('//div[@class="pageNumbers"]/span[contains(@class, "pageNum current")]/text()').\
            extract_first()

        if curPage:
            curPage = int(curPage)
            # E.g. https://www.tripadvisor.com/Attractions-g60763-Activities-oa30
            # -a_allAttractions.true-New_York_City_New_York.html
            #preUrl = "http://www.tripadvisor.com/Attractions-g60763-Activities-oa"
            #sufUrl = "-a_allAttractions.true-New_York_City_New_York.html"

            # E.g. https://www.tripadvisor.com/Attractions-g35805-Activities-oa30
            # -a_allAttractions.true-Chicago_Illinois.html
            #preUrl = "https://www.tripadvisor.com/Attractions-g35805-Activities-oa"
            #sufUrl = "-a_allAttractions.true-Chicago_Illinois.html"

            # E.g. https://www.tripadvisor.com/Attractions-g28970-Activities-oa30
            # -a_allAttractions.true-Washington_DC_District_of_Columbia.html
            #preUrl = "https://www.tripadvisor.com/Attractions-g28970-Activities-oa"
            #sufUrl = "-a_allAttractions.true-Washington_DC_District_of_Columbia.html"

            # E.g. https://www.tripadvisor.com/Attractions-g34438-Activities-oa30
            # -a_allAttractions.true-Miami_Florida.html
            #preUrl = "https://www.tripadvisor.com/Attractions-g34438-Activities-oa"
            #sufUrl = "-a_allAttractions.true-Miami_Florida.html"

            # E.g. https://www.tripadvisor.com/Attractions-g60750-Activities-oa30
            # -a_allAttractions.true-San_Diego_California.html
            preUrl = "https://www.tripadvisor.com/Attractions-g60750-Activities-oa"
            sufUrl = "-a_allAttractions.true-San_Diego_California.html"

            # nextPage = curPage + 1
            # Url: (nextPage -1) * 30 = curPage * 30

            nextPageUrl = preUrl + str(curPage * 30) + sufUrl

            # Note that URL link from "Next" button is reset in html source file, which thus cannot be directly used
            # However, we can use nextPage to check if it is the last page
            nextPage = response.xpath('//a[contains(@class, "ui_button nav next")]/@href').extract()

            if nextPage and len(self.poiSet) <= 600:
                yield scrapy.Request(nextPageUrl, callback=self.parse)

    def parseReviewsSum(self, response):
        # Extract the links for reviews
        reviewLinks = response.xpath('//div[@data-test-target="review-title"]/a/@href').extract()

        for link in reviewLinks:
            # Generate absolute url link for each review
            reviewAbsPath = response.urljoin(link)

            if reviewAbsPath not in self.reviewUrl:
                self.reviewUrl.add(reviewAbsPath)

                yield scrapy.Request(reviewAbsPath, callback=self.parseReview)

        #next_page = response.xpath('//link[@rel="next"]/@href').extract()
        nextPage = response.xpath('//a[contains(@class, "ui_button nav next")]/@href').extract()
        #curPage = int(response.xpath('//div[@class="pageNumbers"]/span[contains(@class, "pageNum current")]/text()').
        #              extract_first())
        curPage = response.xpath('//div[@class="pageNumbers"]/span[contains(@class, "pageNum current")]/text()').\
            extract_first()

        if curPage:
            curPage = int(curPage)
            # 5 reviews/Page
            if nextPage and curPage <= 20:
                # meta: Otherwise redirect 301
                poiAbsPath = response.urljoin(nextPage[-1] + "#REVIEWS")

                if poiAbsPath not in self.poiUrl:
                    self.poiUrl.add(poiAbsPath)

                    yield scrapy.Request(poiAbsPath, callback=self.parseReviewsSum)

    def parseReview(self, response):
        # Name of PoI
        poiName = response.xpath('//div[@class="ui_header attraction_name"]/text()').extract_first()

        # Categories
        categories = response.xpath('//span[@class="is-hidden-mobile header_detail attractionCategories"]'
                                    '/div[@class="detail"]/a/text()').extract()

        categories = ", ".join(categories)

        # User Code
        userCode = response.xpath('//div[@class="info_text"]/div/text()').extract_first()

        # Review title
        reviewTitle = response.xpath('//h1[@id="HEADING"]/text()').extract_first()

        # Rating "ui_bubble_rating bubble_30" / 50 => split bubble_30 => split 30
        # rate = response.xpath('//div[@class="reviewSelector"]/div/div/span/@class').extract_first().\
        #    split()[1].split("_")[1]
        rate = response.xpath('//div[@class="reviewSelector"]/div/div/span/@class').extract_first()

        if rate:
            rate = rate.split()[1].split("_")[1]
        else:
            rate = -1

        # Content of reviews
        # reviewContent = response.xpath('//p[@class="partial_entry"]/span[@class="fullText "]/text()').extract()[-1]
        reviewContent = response.xpath('//p[@class="partial_entry"]/span[@class="fullText "]/text()').extract()

        if reviewContent:
            reviewContent = reviewContent[-1]
        else:
            reviewContent = " "

        # Date of experience
        dateExp = response.xpath('//div[contains(@class, "reviews_stay_date")]/text()').extract_first()

        if dateExp:
            dateExp = dateExp.strip()
        else:
            dateExp = " "

        yield {
            'place':        poiName.encode("utf-8"),
            'categories':   categories.encode("utf-8"),
            'user code':    userCode.encode("utf-8"),
            'title':        reviewTitle.encode("utf-8"),
            'review':       reviewContent.encode("utf-8"),
            'rating':       int(rate),
            'date':         dateExp.encode("utf-8")
        }